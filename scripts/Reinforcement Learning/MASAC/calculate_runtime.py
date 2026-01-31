"""
Calculate actual runtime from log file timestamps
"""
from datetime import datetime, timedelta

# Parse timestamps from filenames
isaac_sim_rrt_start = datetime.strptime("20260116_050620", "%Y%m%d_%H%M%S")
rrt_viz_start = datetime.strptime("20260116_062030", "%Y%m%d_%H%M%S")
astar_start = datetime.strptime("20260116_062047", "%Y%m%d_%H%M%S")

# Calculate durations based on when the next test started
# (assuming minimal gap between tests, with 3 second wait in batch file)
isaac_duration = rrt_viz_start - isaac_sim_rrt_start - timedelta(seconds=3)
rrt_viz_duration = astar_start - rrt_viz_start
# Assume A* took similar time to RRT Viz (both native Python, similar episode counts)
astar_duration = rrt_viz_duration

print("=" * 80)
print("ACTUAL RUNTIME ANALYSIS - 50 Episodes Each")
print("=" * 80)
print()
print("Note: Durations calculated from file timestamps")
print("      Isaac Sim duration = time until RRT Viz started (minus 3s wait)")
print("      RRT Viz duration = time until A* started")
print("      A* duration = assumed same as RRT Viz (both native Python)")
print()

print(f"1. Isaac Sim RRT (50 episodes):")
print(f"   Start:    {isaac_sim_rrt_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   End:      ~{rrt_viz_start.strftime('%Y-%m-%d %H:%M:%S')} (approx)")
print(f"   Duration: {isaac_duration}")
print(f"   Minutes:  {isaac_duration.total_seconds() / 60:.2f} min")
print(f"   Hours:    {isaac_duration.total_seconds() / 3600:.2f} hours")
print()

print(f"2. RRT Viz (50 episodes):")
print(f"   Start:    {rrt_viz_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   End:      ~{astar_start.strftime('%Y-%m-%d %H:%M:%S')} (approx)")
print(f"   Duration: {rrt_viz_duration}")
print(f"   Minutes:  {rrt_viz_duration.total_seconds() / 60:.2f} min")
print(f"   Seconds:  {rrt_viz_duration.total_seconds():.2f} sec")
print()

astar_end_approx = astar_start + astar_duration
print(f"3. A* (50 episodes):")
print(f"   Start:    {astar_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   End:      ~{astar_end_approx.strftime('%Y-%m-%d %H:%M:%S')} (estimated)")
print(f"   Duration: {astar_duration} (estimated, same as RRT Viz)")
print(f"   Minutes:  {astar_duration.total_seconds() / 60:.2f} min")
print(f"   Seconds:  {astar_duration.total_seconds():.2f} sec")
print()

# Calculate total time for the batch run
batch_start = isaac_sim_rrt_start
batch_end = astar_end_approx
total_batch_duration = batch_end - batch_start

print("=" * 80)
print("TOTAL BATCH RUN TIME (All 3 planners, 50 episodes each)")
print("=" * 80)
print(f"Start:    {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End:      {batch_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {total_batch_duration}")
print(f"Minutes:  {total_batch_duration.total_seconds() / 60:.2f} min")
print(f"Hours:    {total_batch_duration.total_seconds() / 3600:.2f} hours")
print()

# Extrapolate to 1000 episodes
print("=" * 80)
print("ESTIMATED TIME FOR 1000 EPISODES (20x scaling)")
print("=" * 80)
print()

isaac_1000 = isaac_duration * 20
rrt_viz_1000 = rrt_viz_duration * 20
astar_1000 = astar_duration * 20
total_1000 = isaac_1000 + rrt_viz_1000 + astar_1000

print(f"1. Isaac Sim RRT (1000 episodes):")
print(f"   Estimated: {isaac_1000}")
print(f"   Hours:     {isaac_1000.total_seconds() / 3600:.2f} hours")
print(f"   Days:      {isaac_1000.total_seconds() / 86400:.2f} days")
print()

print(f"2. RRT Viz (1000 episodes):")
print(f"   Estimated: {rrt_viz_1000}")
print(f"   Hours:     {rrt_viz_1000.total_seconds() / 3600:.2f} hours")
print(f"   Minutes:   {rrt_viz_1000.total_seconds() / 60:.2f} min")
print()

print(f"3. A* (1000 episodes):")
print(f"   Estimated: {astar_1000}")
print(f"   Hours:     {astar_1000.total_seconds() / 3600:.2f} hours")
print(f"   Minutes:   {astar_1000.total_seconds() / 60:.2f} min")
print()

print("=" * 80)
print("TOTAL ESTIMATED TIME FOR 1000 EPISODES (All 3 planners)")
print("=" * 80)
print(f"Total:    {total_1000}")
print(f"Hours:    {total_1000.total_seconds() / 3600:.2f} hours")
print(f"Days:     {total_1000.total_seconds() / 86400:.2f} days")
print()
print("=" * 80)

