from typing import List

from utils.eval_utils import evaluate_episodes, record_video_for_run, plot_eval_curves, parse_args, resolve_algos, PROJECT_ROOT

def main() -> None:
    args = parse_args()

    algos = resolve_algos(args.algo)
    seeds: List[int] = args.seeds
    shaping_name: str = args.shaping
    checkpoint_type: str = args.checkpoint
    n_episodes: int = args.n_episodes
    render: str = args.render

    print("====================================")
    print(" Evaluation configuration")
    print("====================================")
    print(f"Algorithms    : {algos}")
    print(f"Seeds         : {seeds}")
    print(f"Reward shaping: {shaping_name}")
    print(f"Checkpoint    : {checkpoint_type}")
    print(f"Episodes/run  : {n_episodes}")
    print(f"Render mode   : {render}")
    print(f"Plot curves   : {args.plot}")
    print("====================================\n")

    # 1) Evaluation / video
    for algo in algos:
        for seed in seeds:
            print(f"\n=== Evaluating {algo.upper()} (seed={seed}, shaping={shaping_name}) ===")

            if render == "video":
                video_path = record_video_for_run(
                    algo_name=algo,
                    seed=seed,
                    shaping_name=shaping_name,
                    checkpoint_type=checkpoint_type,
                )
                if video_path is not None:
                    print(f"[INFO] Video saved at: {video_path}")
            else:
                # render_mode for evaluate_episodes
                if render == "human":
                    render_mode = "human"
                else:
                    render_mode = None

                returns = evaluate_episodes(
                    algo_name=algo,
                    seed=seed,
                    shaping_name=shaping_name,
                    checkpoint_type=checkpoint_type,
                    n_episodes=n_episodes,
                    render_mode=render_mode,
                )
                mean_ret = sum(returns) / len(returns)
                print(f"[RESULT] {algo.upper()} seed={seed}, "
                      f"mean return over {n_episodes} episodes: {mean_ret:.2f}")

    # 2) Plot learning curves (from training logs) if requested
    if args.plot:
        plots_dir = PROJECT_ROOT / "plots"
        title_suffix = f"checkpoint={checkpoint_type}"
        plot_eval_curves(
            algos=algos,
            seeds=seeds,
            shaping_name=shaping_name,
            output_dir=plots_dir,
            title_suffix=title_suffix,
        )

    print("\nAll evaluation runs finished.")


if __name__ == "__main__":
    main()
