from utils.algos import train_one_run
from utils.train_utils import parse_args, List, resolve_algos, COMMON_CFG



def main() -> None:
    args = parse_args()

    algos = resolve_algos(args.algo)
    seeds: List[int] = args.seeds
    shaping_name: str = args.shaping

    print("====================================")
    print(" Training configuration")
    print("====================================")
    print(f"Algorithms    : {algos}")
    print(f"Seeds         : {seeds}")
    print(f"Reward shaping: {shaping_name}")
    print(f"Timesteps/run : {args.total_timesteps}")
    print("====================================\n")

    # Override total_timesteps in COMMON_CFG at runtime (simple approach)
    COMMON_CFG["total_timesteps"] = args.total_timesteps

    for algo in algos:
        for seed in seeds:
            print(f"\n=== Training {algo.upper()} (seed={seed}, shaping={shaping_name}) ===")
            train_one_run(
                algo_name=algo,
                seed=seed,
                shaping_name=shaping_name,
            )
    print("\nAll training runs finished.")


if __name__ == "__main__":
    main()
