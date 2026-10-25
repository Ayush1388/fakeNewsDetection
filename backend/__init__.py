import os

# Small shared hosts report many CPU cores but give a fraction of one; letting
# BLAS / OpenMP spawn a thread per reported core makes training crawl there.
# Must be set before numpy / scikit-learn are imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
