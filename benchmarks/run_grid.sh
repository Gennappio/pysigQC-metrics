#!/usr/bin/env bash
# Full experiment grid of the scalability study. Every run is cached as a JSON
# under benchmark_results/, so the script can be interrupted and resumed.
#
#   PY=.venv/bin/python benchmarks/run_grid.sh            # everything
#   PY=.venv/bin/python benchmarks/run_grid.sh main       # one stage
#
# Stages: data main density chunks cache fast real v2
#
# main/density/chunks/cache measure the sparse backends *before* the exact fast
# paths (--no-fast-paths), so that every gain stays attributable; 'fast' adds them.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
GEN="$PY benchmarks/generate_singlecell.py"
BENCH="$PY benchmarks/benchmark_pipeline.py"
NOFAST="--no-fast-paths"
STAGES="${*:-data main density chunks cache fast real v2}"
SPARSE6="scipy_csr scipy_csc h5ad_csr h5ad_csc zarr_csr zarr_csc"
ALL7="dense scipy_csr scipy_csc h5ad_csr h5ad_csc zarr_csr zarr_csc"

for stage in $STAGES; do
case "$stage" in
data)
    $GEN --cells 1000 --genes 2000 --density 0.05 --derive
    $GEN --cells 10000 50000 100000 250000 1000000 --genes 20000 --density 0.05 --derive
    $GEN --cells 100000 1000000 --genes 20000 --density 0.01 0.10 --derive
    # Layout / chunking / compression variants (experiment F/G chunk benchmark)
    for n in 100000 1000000; do
        $GEN --cells $n --genes 20000 --density 0.05 --derive \
            zarr_csr_c1k zarr_csr_c10k zarr_csr_c50k \
            zarr_csc_g10 zarr_csc_g100 zarr_csc_g1000 \
            zarr_csr_raw zarr_csc_raw h5ad_csr_gzip h5ad_csc_gzip
    done
    ;;
main)      # A-G at 5% density, all sizes
    $BENCH --dataset sc_1000x2000_d05 --backends $ALL7 $NOFAST
    for n in 10000 50000 100000 250000 1000000; do
        $BENCH --dataset sc_${n}x20000_d05 --backends $ALL7 $NOFAST
    done
    ;;
density)   # A-G at 1% and 10%
    for d in 01 10; do for n in 100000 1000000; do
        $BENCH --dataset sc_${n}x20000_d${d} --backends $ALL7 $NOFAST
    done; done
    ;;
chunks)    # Zarr chunk size, compression, dask read_lazy
    for n in 100000 1000000; do
        $BENCH --dataset sc_${n}x20000_d05 --backends \
            zarr_csr_c1k zarr_csr_c10k zarr_csr_c50k \
            zarr_csc_g10 zarr_csc_g100 zarr_csc_g1000 \
            zarr_csr_raw zarr_csc_raw h5ad_csr_gzip h5ad_csc_gzip \
            zarr_csr+dask zarr_csc+dask $NOFAST
    done
    ;;
cache)     # run_pipeline() with and without the shared DatasetStatsCache
    for n in 100000 1000000; do
        for b in scipy_csr scipy_csc h5ad_csr h5ad_csc zarr_csr zarr_csc; do
            $BENCH --dataset sc_${n}x20000_d05 --backends $b --mode pipeline --tag pipeline $NOFAST
            $BENCH --dataset sc_${n}x20000_d05 --backends $b --mode pipeline --share-cache --tag shared_cache $NOFAST
        done
    done
    ;;
fast)      # + exact fast median / sparse-aware ranking, without and with the cache
    for n in 100000 250000 1000000; do
        $BENCH --dataset sc_${n}x20000_d05 --backends $SPARSE6 --mode pipeline --tag fastpaths
        $BENCH --dataset sc_${n}x20000_d05 --backends $SPARSE6 --mode pipeline --share-cache --tag fastpaths_cache
    done
    # per-module profile of the final state, and the chunk-alignment check
    $BENCH --dataset sc_1000000x20000_d05 --backends $SPARSE6 --tag fastpaths_modules
    $BENCH --dataset sc_1000000x20000_d05 --backends zarr_csr_c50k --chunk-nnz 100000000 --tag aligned $NOFAST
    ;;
real)      # public datasets (see prepare_real.py)
    for d in real_covid_lung_116k real_onek1k_1p25M; do
        [ -d benchmarks/data/$d ] || continue
        $BENCH --dataset $d --backends dense $SPARSE6 $NOFAST
        $BENCH --dataset $d --backends $SPARSE6 --mode pipeline --share-cache --tag fastpaths_cache
        [ -e benchmarks/data/$d/h5ad_orig.h5ad ] && $BENCH --dataset $d --backends h5ad_orig --mode pipeline --share-cache --tag fastpaths_cache
    done
    ;;
v2)        # second optimisation round: cache on by default, memory work, persisted statistics
    for n in 100000 250000 1000000; do
        $BENCH --dataset sc_${n}x20000_d05 --backends $SPARSE6 --mode pipeline --share-cache --tag v2
    done
    # real working set (macOS keeps freed large blocks resident otherwise)
    $BENCH --dataset sc_1000000x20000_d05 --backends $SPARSE6 --mode pipeline --share-cache --no-malloc-cache --tag v2_ws
    # the DataFrame path through today's package
    for n in 10000 50000 100000 250000; do
        $BENCH --dataset sc_${n}x20000_d05 --backends dataframe --mode pipeline --share-cache --tag v2
    done
    # persisted per-gene statistics: first run builds the file, second one reuses it
    for d in sc_1000000x20000_d05 real_onek1k_1p25M; do
        [ -d benchmarks/data/$d ] || continue
        for b in $SPARSE6; do
            sc=benchmarks/data/$d/stats_cache/$b
            $BENCH --dataset $d --backends $b --mode pipeline --share-cache --stats-cache $sc --tag v2_stats_build
            $BENCH --dataset $d --backends $b --mode pipeline --share-cache --stats-cache $sc --tag v2_stats_reuse
        done
    done
    for d in real_covid_lung_116k real_onek1k_1p25M; do
        [ -d benchmarks/data/$d ] || continue
        $BENCH --dataset $d --backends $SPARSE6 --mode pipeline --share-cache --tag v2
    done
    ;;
*) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
done
$BENCH --aggregate
