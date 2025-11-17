# --------------------------
# Example usage (run these in your REPL/script; do not put Pkg.add in the module)
# --------------------------
using Pkg
Pkg.add(["CSV","DataFrames","Optim"])  # run once in your environment
#
using .Rust_replication
#
out = Rust_replication.run_rust1987("group_4.csv";
                state_col = :mileage,        # use raw miles
                decision_col = :decision,
                x_is_bin = false)            # tell the loader explicitly

# # Optional: sanity check the likelihood at Rust’s published params on the SAME inputs
# df    = CSV.read("group_4.csv", DataFrame)
# model = Rust_replication.build_model()
# println("nll at Rust params (mileage-as-miles): ",
#         Rust_replication.nll_at_rust(df, model; state_col=:mileage, x_is_bin=false))