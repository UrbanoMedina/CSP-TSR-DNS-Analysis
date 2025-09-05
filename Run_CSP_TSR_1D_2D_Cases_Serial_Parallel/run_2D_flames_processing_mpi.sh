#!/bin/bash
# Wrapper script to run MPI jobs with safe threading settings

show_help() {
    echo "Usage: $0 [NP] [SCRIPT] [THREADS]"
    echo
    echo "Arguments:"
    echo "  NP        Number of MPI processes (default: 32)"
    echo "  SCRIPT    Python script to run (default: 2D_Flames_Parametric_Analysis_MPI.py)"
    echo "  THREADS   Number of threads per MPI rank (default: 1 = pure MPI)"
    echo
    echo "Examples:"
    echo "  $0                # Run 32 ranks, each 1 thread (pure MPI)"
    echo "  $0 10             # Run 10 ranks, each 1 thread"
    echo "  $0 8 myscript.py  # Run 8 ranks, each 1 thread"
    echo "  $0 8 myscript.py 4 # Run 8 ranks, each with 4 threads (hybrid MPI+OpenMP)"
    echo
    echo "Note: Total cores used = NP × THREADS. Make sure it does not exceed 32 on this machine."
    exit 0
}

# If user asks for help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
fi

# ===== CONFIGURE DEFAULTS =====
NP=${1:-32}                               # Number of MPI processes
SCRIPT=${2:-2D_Flames_Parametric_Analysis_MPI.py}  # Python script
THREADS=${3:-1}                           # Threads per process (default=1, pure MPI)
# ==============================

# Set environment variables to control threading
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export VECLIB_MAXIMUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

# Run with MPI, binding ranks to physical cores and spreading across sockets
mpirun -np $NP \
    --bind-to core \
    --map-by socket \
    -x OMP_NUM_THREADS \
    -x OPENBLAS_NUM_THREADS \
    -x MKL_NUM_THREADS \
    -x VECLIB_MAXIMUM_THREADS \
    -x NUMEXPR_NUM_THREADS \
    python3 $SCRIPT
