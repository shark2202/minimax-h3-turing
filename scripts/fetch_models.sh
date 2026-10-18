#!/usr/bin/env bash
# Manifest-driven model fetcher for this machine, with resume (断点续传).
#
#   scripts/fetch_models.sh --list                 # 清单 + 本机状态
#   scripts/fetch_models.sh --check                # 只校验：字节数 + sha256
#   scripts/fetch_models.sh                        # 缺什么下什么（已就位且校验通过的跳过）
#   scripts/fetch_models.sh --only 9b --jobs 12 --mirror hf-mirror
#   scripts/fetch_models.sh --only vae --force     # 忽略已就位，重新下发
#
# 断点续传怎么做的：每个文件切成 --jobs 片，落在 <dest>.parts/partNN；每片反复用
#
#     curl -C - -r <start>-<end> -o partNN <url>
#
# 请求。-C - 会把 Range 起点按**已落盘的字节数**偏移，所以中断、重跑、换网络都只是接着写；
# 片本身超长会被截回期望长度（防重试叠加），换过 URL 会丢弃旧片重来（见 url.txt 标记）。
# 装前校验：拼好的文件必须**字节数与 sha256 都对**才 mv 到目标；不符 exit 2 并保留分片。
#
# 纪律（本机教训，RUNBOOK §4.5）：**下载期间不要渲染**（并发负载会让采样出 NaN、成片全黑）。
# 检测到 ComfyUI / llama-server 在跑会大声警告——不拦，因为你自己知道在干什么。
#
# Exit: 0 = 全部就位且校验通过；1 = 有文件没下完/下载失败；2 = sha256 不符。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-$HERE/../models/MANIFEST.tsv}"
JOBS=8
MODE="fetch"          # fetch | check | list
ONLY=""
MIRROR="asis"         # asis | hf-mirror | official
FORCE=0
QUIET=0

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --list) MODE="list" ;;
        --check) MODE="check" ;;
        --only) ONLY="$2"; shift ;;
        --jobs|--parts) JOBS="$2"; shift ;;
        --mirror) MIRROR="$2"; shift ;;
        --manifest) MANIFEST="$2"; shift ;;
        --force) FORCE=1 ;;
        --quiet) QUIET=1 ;;
        -h|--help) usage 0 ;;
        *) usage 2 ;;
    esac
    shift
done

[ -f "$MANIFEST" ] || { echo "no manifest: $MANIFEST" >&2; exit 1; }
command -v curl >/dev/null || { echo "curl required" >&2; exit 1; }

rewrite_url() {  # $1 = url
    case "$MIRROR" in
        hf-mirror) printf '%s' "${1/huggingface.co/hf-mirror.com}" ;;
        official)  printf '%s' "${1/hf-mirror.com/huggingface.co}" ;;
        *)         printf '%s' "$1" ;;
    esac
}

# One-line status for a manifest row: OK | MISSING | BAD-size | BAD-sha
row_status() {  # <bytes> <sha256> <dest>
    local bytes="$1" sha="$2" dest="$3"
    [ -f "$dest" ] || { printf 'MISSING'; return; }
    [ "$(stat -c %s "$dest")" = "$bytes" ] || { printf 'BAD-size'; return; }
    if [ "$MODE" = "list" ]; then printf 'present'; return; fi   # --list 不做全量哈希
    if printf '%s  %s\n' "$sha" "$dest" | sha256sum -c --status 2>/dev/null; then
        printf 'OK'
    else
        printf 'BAD-sha'
    fi
}

fetch_part() {  # <url> <dest> <total> <jobs> <idx>
    local url="$1" dest="$2" total="$3" jobs="$4" idx="$5"
    local span=$(( (total + jobs - 1) / jobs ))
    local base end
    base=$(( idx * span ))
    end=$(( base + span - 1 ))
    [ "$end" -ge "$total" ] && end=$(( total - 1 ))
    [ "$base" -ge "$total" ] && return 0            # 片数多于字节（小文件）
    local want=$(( end - base + 1 ))
    local part="$parts_dir/part$(printf '%02d' "$idx")"
    local have attempt=0
    while :; do
        have=$(stat -c %s "$part" 2>/dev/null || echo 0)
        if [ "$have" -gt "$want" ]; then
            echo "  part$idx 超出期望长度（$have > $want）：源可能不支持 Range" >&2
            truncate -s "$want" "$part"
            return 1
        fi
        [ "$have" -ge "$want" ] && return 0
        attempt=$(( attempt + 1 ))
        [ "$attempt" -gt 25 ] && { echo "  part$idx: 25 次尝试仍未完成" >&2; return 1; }
        # 自己算续传起点（base + 已落盘字节），**不要**用 curl -C - 配 --range：
        # 实测那组合会把 range 的结尾丢掉、发成 bytes=<offset>-，等于每片都从断点重下到 EOF。
        # 也**不要**加 curl --retry：重试会从 range 头重来，而我们已经 append 过一段 → 会重复写入。
        # 每次尝试都是"请求 [base+have, end]、追加到片上"，中断/失败/重跑都只是接着写。
        curl --silent --show-error --location --fail \
             --connect-timeout 20 --max-time 3600 \
             --range "$(( base + have ))-${end}" "$url" >> "$part" || {
            sleep $(( attempt < 5 ? 3 : 15 ))
            continue
        }
    done
}

fetch_one() {  # <bytes> <sha> <dest> <url> <role>
    local bytes="$1" sha="$2" dest="$3" url="$4" role="$5"
    parts_dir="${dest}.parts"
    mkdir -p "$(dirname "$dest")" "$parts_dir"
    # 换了源就不能用旧分片（内容可能不是同一个对象）
    if [ -f "$parts_dir/url.txt" ] && [ "$(cat "$parts_dir/url.txt")" != "$url" ]; then
        echo "  源变了，丢弃旧分片重来"
        rm -rf "${parts_dir:?}"/*
    fi
    printf '%s\n' "$url" > "$parts_dir/url.txt"

    echo "[fetch] $role"
    echo "        $dest"
    echo "        ${bytes} bytes, ${JOBS} 片, 源 ${url}"
    local idx=0
    while [ "$idx" -lt "$JOBS" ]; do
        fetch_part "$url" "$dest" "$bytes" "$JOBS" "$idx" &
        idx=$(( idx + 1 ))
    done
    wait

    local got
    idx=0
    while [ "$idx" -lt "$JOBS" ]; do
        local p="$parts_dir/part$(printf '%02d' "$idx")"
        local span=$(( (bytes + JOBS - 1) / JOBS ))
        local base end
        base=$(( idx * span ))
        end=$(( base + span - 1 ))
        [ "$end" -ge "$bytes" ] && end=$(( bytes - 1 ))
        if [ "$base" -lt "$bytes" ]; then
            if [ "$(stat -c %s "$p" 2>/dev/null || echo 0)" != "$(( end - base + 1 ))" ]; then
                echo "  分片 $idx 不完整，重跑本脚本可续传" >&2
                return 1
            fi
        fi
        idx=$(( idx + 1 ))
    done

    cat "$parts_dir"/part* > "$dest.tmp"
    got=$(stat -c %s "$dest.tmp")
    if [ "$got" != "$bytes" ]; then
        echo "  拼装后字节数 $got != $bytes，保留分片待续" >&2
        rm -f "$dest.tmp"
        return 1
    fi
    if ! printf '%s  %s\n' "$sha" "$dest.tmp" | sha256sum -c --status; then
        echo "  sha256 不符（$dest.tmp）；分片保留在 $parts_dir，可重跑续传" >&2
        rm -f "$dest.tmp"
        return 2
    fi
    mv -f "$dest.tmp" "$dest"
    rm -rf "$parts_dir"
    echo "  ✓ 已就位并校验通过"
    return 0
}

if [ "$MODE" = "fetch" ] && [ "$FORCE" -eq 0 ]; then
    if pgrep -f "main[.]py --listen" >/dev/null 2>&1 || pgrep -f "[l]lama-server" >/dev/null 2>&1; then
        echo "⚠️  检测到 ComfyUI / llama-server 在跑 —— 下载期间别渲染（RUNBOOK §4.5：并发负载会让采样出 NaN、成片全黑）" >&2
    fi
fi

printf '%-9s %-13s %-11s %s\n' SHA256 SIZE STATUS DEST
rc=0
matched=0
while IFS='|' read -r sha bytes dest url alt role; do
    case "${sha:-}" in ''|'#'*) continue ;; esac
    if [ -z "${bytes:-}" ] || [ -z "${dest:-}" ] || [ -z "${url:-}" ]; then
        echo "清单某行不是 6 段竖线分隔：sha|bytes|dest|url|alt|role -> $sha" >&2
        exit 1
    fi
    if [ -n "$ONLY" ] && ! printf '%s' "$dest $role" | grep -qi -- "$ONLY"; then continue; fi
    matched=$(( matched + 1 ))
    url="$(rewrite_url "$url")"
    status="$(row_status "$bytes" "$sha" "$dest")"
    printf '%-9s %-13s %-11s %s\n' "${sha:0:8}" "$bytes" "$status" "$dest"

    [ "$MODE" = "list" ] && continue
    if [ "$MODE" = "check" ]; then
        case "$status" in OK|present) ;; BAD-sha) rc=2 ;; *) [ "$rc" -eq 0 ] && rc=1 ;; esac
        continue
    fi
    case "$status" in
        OK) [ "$QUIET" -eq 0 ] && echo "  跳过（已就位 + sha256 通过）" ;;
        *)  fetch_one "$bytes" "$sha" "$dest" "$url" "$role" || {
                code=$?
                [ "$code" -eq 2 ] && rc=2 || { [ "$rc" -ne 2 ] && rc=1; }
            } ;;
    esac
done < "$MANIFEST"

[ "$matched" -eq 0 ] && { echo "没有匹配的行（--only $ONLY）" >&2; exit 1; }
if [ "$MODE" = "fetch" ]; then
    case "$rc" in
        0) echo "=> 全部就位且校验通过" ;;
        1) echo "=> 还有没下完的：重跑本脚本即从断点继续（分片在 <dest>.parts/）" ;;
        2) echo "=> 有文件 sha256 不符：分片保留，可重跑；仍不符说明源上的对象变了" ;;
    esac
fi
exit "$rc"
