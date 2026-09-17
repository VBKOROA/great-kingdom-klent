#!/bin/bash
# Sourced by monitor.sh. No host-wide counters are used for container CPU usage.

decode_mount_path() {
    printf '%b' "$1"
}

# Resolve membership against mount roots, including cgroup namespace mounts.
find_cgroup_dir() {
    local controller=$1 membership='' hierarchy controllers path
    while IFS=: read -r hierarchy controllers path; do
        if { [ "$controller" = v2 ] && [ "$hierarchy" = 0 ] && [ -z "$controllers" ]; } ||
           { [ "$controller" != v2 ] && [[ ",$controllers," == *",$controller,"* ]]; }; then
            membership=$path
            break
        fi
    done < /proc/self/cgroup
    [ -n "$membership" ] || return 1

    local line before after root mountpoint fs source options relative candidate
    while IFS= read -r line; do
        before=${line%% - *}
        after=${line#* - }
        read -r fs source options _ <<< "$after"
        if [ "$controller" = v2 ]; then
            [ "$fs" = cgroup2 ] || continue
        else
            [ "$fs" = cgroup ] && [[ ",$options," == *",$controller,"* ]] || continue
        fi
        read -r _ _ _ root mountpoint _ <<< "$before"
        root=$(decode_mount_path "$root")
        mountpoint=$(decode_mount_path "$mountpoint")
        if [ "$membership" = "$root" ]; then
            relative=''
        elif [ "$root" = / ]; then
            relative=$membership
        elif [[ "$membership" == "$root/"* ]]; then
            relative=${membership#"$root"}
        elif [ "$membership" = / ]; then
            relative=''
        else
            continue
        fi
        candidate=${mountpoint%/}$relative
        if [ -d "$candidate" ]; then
            CGROUP_DIR=$candidate
            CGROUP_MOUNT=$mountpoint
            return 0
        fi
    done < /proc/self/mountinfo
    return 1
}

init_cpu_monitor() {
    CPU_KIND=unavailable
    if find_cgroup_dir v2 && [ -r "$CGROUP_DIR/cpu.stat" ]; then
        CPU_KIND=v2
        CPU_USAGE_DIR=$CGROUP_DIR
        CPU_LIMIT_DIR=$CGROUP_DIR
        CPU_LIMIT_MOUNT=$CGROUP_MOUNT
    elif find_cgroup_dir cpuacct && [ -r "$CGROUP_DIR/cpuacct.usage" ]; then
        CPU_KIND=v1
        CPU_USAGE_DIR=$CGROUP_DIR
        CPU_LIMIT_DIR=''
        CPU_LIMIT_MOUNT=''
        if find_cgroup_dir cpu; then
            CPU_LIMIT_DIR=$CGROUP_DIR
            CPU_LIMIT_MOUNT=$CGROUP_MOUNT
        fi
    fi
}

cpu_capacity() {
    # Affinity accounts for cpuset restrictions. Also respect visible ancestor quotas.
    local capacity dir quota period
    capacity=$(nproc 2>/dev/null) || return 1
    dir=$CPU_LIMIT_DIR
    while [ -n "$dir" ]; do
        quota=''
        period=''
        if [ "$CPU_KIND" = v2 ] && [ -r "$dir/cpu.max" ]; then
            read -r quota period < "$dir/cpu.max"
        elif [ "$CPU_KIND" = v1 ] && [ -r "$dir/cpu.cfs_quota_us" ] && [ -r "$dir/cpu.cfs_period_us" ]; then
            read -r quota < "$dir/cpu.cfs_quota_us"
            read -r period < "$dir/cpu.cfs_period_us"
        fi
        capacity=$(awk -v cap="$capacity" -v q="$quota" -v p="$period" \
            'BEGIN { if (q + 0 > 0 && p + 0 > 0 && q/p < cap) cap=q/p; printf "%.6f", cap }')
        [ "$dir" = "$CPU_LIMIT_MOUNT" ] && break
        [ "$dir" = / ] && break
        dir=${dir%/*}
        [ -n "$dir" ] || dir=/
    done
    printf '%s\n' "$capacity"
}

cpu_snapshot() {
    local elapsed usage
    read -r elapsed _ < /proc/uptime
    case "$CPU_KIND" in
        v2) usage=$(awk '$1 == "usage_usec" {print $2; found=1} END {if (!found) exit 1}' "$CPU_USAGE_DIR/cpu.stat") || return 1 ;;
        v1) usage=$(awk '{printf "%.0f", $1 / 1000}' "$CPU_USAGE_DIR/cpuacct.usage") || return 1 ;;
        *) return 1 ;;
    esac
    printf '%s %s\n' "$elapsed" "$usage"
}

format_cpu_usage() {
    awk -v prev_time="$1" -v prev_usage="$2" -v now="$3" -v usage="$4" -v cap="$5" '
        BEGIN {
            dt=now-prev_time; du=usage-prev_usage
            if (dt <= 0 || du < 0 || cap <= 0) {
                print "CPU 사용량 (cgroup): 측정 대기"
                exit
            }
            busy=du/1000000/dt
            printf "CPU 사용량 (cgroup): %.1f%% of capacity, %.2f/%.2f vCPU busy\n", 100*busy/cap, busy, cap
        }'
}
