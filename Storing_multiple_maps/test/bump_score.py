# 防止预分配内存
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
import jax
from Network_Multiple_Maps_wyl import Place_net, Grid_net
import time
import os

# 设置工作路径
work_dir = f'/home/yulingwu/CCS_place_grid/Storing_multiple_maps/test/result_20250709/grid_module_test'
os.makedirs(work_dir, exist_ok=True)

W = 0
place_num = 1000

# grid_nums = [210, 105, 70, 53, 42, 35, 26, 23, 21]
# module_num = [1, 2, 3, 4, 5, 6, 8, 9, 10]
grid_nums = [20]
module_num = [10]

for grid_num, module_num in zip(grid_nums, module_num):
  print(f"place_num: {place_num}, grid_num: {grid_num}, module_num: {module_num}")
  # 设置次级目录
  sub_dir = os.path.join(work_dir, f'place_{place_num}_grid_{grid_num}_module_{module_num}')
  os.makedirs(sub_dir, exist_ok=True)
  os.chdir(sub_dir)

  # 记录开始时间
  start_time = time.time()

  # Define constants
  z_min, z_max = 0, 20
  a_p = 0.5
  Spacing = bm.linspace(6, 20, module_num)
  num_map = 30
  map_num_all = bm.arange(num_map) + 2 # 2-17
  simulaiton_num = 100

  # Preallocate arrays
  bump_score_orginal = bm.zeros(num_map)
  bump_score_others = bm.zeros(num_map)
  bump_score_diff = bm.zeros(num_map)
  bump_score_orginal_std = bm.zeros(num_map)
  bump_score_others_std = bm.zeros(num_map)
  bump_score_diff_std = bm.zeros(num_map)


  # Preallocate arrays
  bump_score_orginal_only = bm.zeros(num_map)
  bump_score_others_only = bm.zeros(num_map)
  bump_score_diff_only = bm.zeros(num_map)
  bump_score_orginal_std_only = bm.zeros(num_map)
  bump_score_others_std_only = bm.zeros(num_map)
  bump_score_diff_std_only = bm.zeros(num_map)

  # Main loop
  for i, map_num in enumerate(map_num_all):
    bump_score_orginal_monte = bm.Variable(bm.zeros(simulaiton_num))
    bump_score_others_monte = bm.Variable(bm.zeros(simulaiton_num))
    bump_score_diff_monte = bm.Variable(bm.zeros(simulaiton_num))

    bump_score_orginal_monte_only = bm.Variable(bm.zeros(simulaiton_num))
    bump_score_others_monte_only = bm.Variable(bm.zeros(simulaiton_num))
    bump_score_diff_monte_only = bm.Variable(bm.zeros(simulaiton_num))

    # for j in range(simulaiton_num):
    def update_func(j):
      Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num,
                            noise_stre=0.5)
      maps = Place_cell.map
      place_index = Place_cell.place_index


      Place_cell_only = Place_net(z_min=z_min, z_max=z_max, map_num=map_num, neuron_num=place_num, place_num=place_num,
                            noise_stre=0.5)

      # Grid module list initialization
      Gird_module_list = [Grid_net(L=Spacing[module], maps=maps, place_index=place_index, neuron_num=grid_num, J0=5, W0=W,
                                  a_g=a_p / Spacing[module] * 2 * bm.pi) for module in range(module_num)]
      map_index = 0

      def run_net(indices, loc, input_stre):
        r_hpc = Place_cell.r
        output = bm.zeros(place_num, )
        for Grid_cell in Gird_module_list:
          Grid_cell.step_run(indices, r_hpc=r_hpc, loc=loc, input_stre=input_stre, map_index=map_index)
          output += Grid_cell.output
        Place_cell.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre, input_g=output)
        Place_cell_only.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre)
        return Place_cell.r.value, Place_cell_only.r.value


      total_time = 20000
      start_time = 1000
      indices = bm.arange(total_time)
      loc = bm.zeros(total_time) + (z_max + z_min) / 2
      input_stre = bm.zeros(total_time)
      input_stre[:start_time] = 10.

      us, us_only = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar=False)
      u = us[-1]
      u_only = us_only[-1]
      loc_num = 100
      loc_candidate = bm.linspace(z_min, z_max, loc_num, endpoint=False)

      def cosine_similarity(a, b): #所有bump score的计算都改成这个
        dot_product = jax.numpy.dot(a, b)
        norm_a = jax.numpy.linalg.norm(a)
        norm_b = jax.numpy.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

      def compute_bump_score(u, loc_candidate, map_num, Place_cell, place_index):
        def body(map_index):
          u_place = u[place_index[map_index]]
          
          def score_func(loc):
            bump = Place_cell.get_bump(map_index, loc) 
            return cosine_similarity(bump, u_place)

          score_candidate = jax.vmap(score_func)(loc_candidate)
          return bm.max(score_candidate)

        bump_score = bm.for_loop(body, (bm.arange(map_num)), progress_bar=False)
        return bump_score
      

      bump_score = compute_bump_score(u, loc_candidate, map_num, Place_cell, place_index) # coupled net
      bump_score_orginal_monte[j] = bump_score[0] 
      bump_score_others_monte[j] = bm.max(bump_score[1:])
      bump_score_diff_monte[j] = bump_score_orginal_monte[j] - bump_score_others_monte[j]


      bump_score_only = compute_bump_score(u_only, loc_candidate, map_num, Place_cell_only, place_index) # only place net
      bump_score_orginal_monte_only[j] = bump_score_only[0]
      bump_score_others_monte_only[j] = bm.max(bump_score_only[1:])
      bump_score_diff_monte_only[j] = bump_score_orginal_monte_only[j] - bump_score_others_monte_only[j]


      # jax.experimental.io_callback(print((i * simulaiton_num + j) / (simulaiton_num * num_map)))
    print('simulaiton trail:', i)
    bm.for_loop(update_func, bm.arange(simulaiton_num), progress_bar=True)
    bump_score_diff_mean = bm.mean(bump_score_diff_monte)
    if bump_score_diff_mean < 0.3:
      max_map_num = map_num - 1
      print(f"Max map number for grid_num {grid_num} is {max_map_num}")
      # break

    bump_score_orginal = bump_score_orginal.at[i].set(bm.mean(bump_score_orginal_monte))
    bump_score_others = bump_score_others.at[i].set(bm.mean(bump_score_others_monte))
    bump_score_diff = bump_score_diff.at[i].set(bump_score_diff_mean)
    bump_score_orginal_std = bump_score_orginal_std.at[i].set(bm.std(bump_score_orginal_monte))
    bump_score_others_std = bump_score_others_std.at[i].set(bm.std(bump_score_others_monte))
    bump_score_diff_std = bump_score_diff_std.at[i].set(bm.std(bump_score_diff_monte))

    bump_score_orginal_only = bump_score_orginal_only.at[i].set(bm.mean(bump_score_orginal_monte_only))
    bump_score_others_only = bump_score_others_only.at[i].set(bm.mean(bump_score_others_monte_only))
    bump_score_diff_only = bump_score_diff_only.at[i].set(bm.mean(bump_score_diff_monte_only))
    bump_score_orginal_std_only = bump_score_orginal_std_only.at[i].set(bm.std(bump_score_orginal_monte_only))
    bump_score_others_std_only = bump_score_others_std_only.at[i].set(bm.std(bump_score_others_monte_only))
    bump_score_diff_std_only = bump_score_diff_std_only.at[i].set(bm.std(bump_score_diff_monte_only))



  plt.figure(figsize=(6, 5))

  # Plot data from the first file
  plt.plot(map_num_all, bump_score_diff_only, label='940 Place cells')
  plt.plot(map_num_all, bump_score_diff, label='800 Place cells with 20*7 grid cells')
  plt.legend()
  plt.errorbar(map_num_all, bump_score_diff_only, yerr=bump_score_diff_std_only, fmt='o', capsize=5)
  plt.errorbar(map_num_all, bump_score_diff, yerr=bump_score_diff_std, fmt='s', capsize=5)

  plt.xlabel('Embedded Map Number')
  plt.ylabel('Bump Score Difference')
  plt.title('Errorbar Plot of Bump Score Differences')
  plt.legend()
  plt.grid(True)

  # Save the plot
  plt.tight_layout()
  plt.savefig('Bump_Score_Difference_Comparison_noshufflephase.png')

  # Create subplots
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8))

  # First subplot: bump_score_orginal and bump_score_others
  ax1.errorbar(map_num_all, bump_score_orginal, yerr=bump_score_orginal_std, fmt='-o', label='Original')
  ax1.errorbar(map_num_all, bump_score_others, yerr=bump_score_others_std, fmt='-o', label='Others')
  ax1.set_title('Error Bar Plot of Bump Scores: Original and Others')
  ax1.set_xlabel('Embedded Map Number')
  ax1.set_ylabel('Bump Score')
  ax1.legend()
  ax1.grid(True)

  # Second subplot: bump_score_diff
  ax2.errorbar(map_num_all, bump_score_diff, yerr=bump_score_diff_std, fmt='-o', label='Difference')
  ax2.set_title('Error Bar Plot of Bump Score Difference')
  ax2.set_xlabel('Embedded Map Number')
  ax2.set_ylabel('Bump Score Difference')
  ax2.legend()
  ax2.grid(True)

  # Save the plot
  plt.tight_layout()
  plt.savefig('bump_score_errorbars_coupled_net_noshufflephase.png')

  # First subplot: bump_score_orginal and bump_score_others
  ax1.errorbar(map_num_all, bump_score_orginal_only, yerr=bump_score_orginal_std_only, fmt='-o', label='Original')
  ax1.errorbar(map_num_all, bump_score_others_only, yerr=bump_score_others_std_only, fmt='-o', label='Others')
  ax1.set_title('Error Bar Plot of Bump Scores: Original and Others')
  ax1.set_xlabel('Embedded Map Number')
  ax1.set_ylabel('Bump Score')
  ax1.legend()
  ax1.grid(True)

  # Second subplot: bump_score_diff
  ax2.errorbar(map_num_all, bump_score_diff_only, yerr=bump_score_diff_std_only, fmt='-o', label='Difference')
  ax2.set_title('Error Bar Plot of Bump Score Difference')
  ax2.set_xlabel('Embedded Map Number')
  ax2.set_ylabel('Bump Score Difference')
  ax2.legend()
  ax2.grid(True)

  # Save the plot
  plt.tight_layout()
  plt.savefig('bump_score_errorbars_place_only.png')

  # Save the data
  np.savez('bump_scores_coupled_net_shufflephase.npz',
          bump_score_orginal=bump_score_orginal,
          bump_score_others=bump_score_others,
          bump_score_diff=bump_score_diff,
          bump_score_orginal_std=bump_score_orginal_std,
          bump_score_others_std=bump_score_others_std,
          bump_score_diff_std=bump_score_diff_std)

  np.savez('bump_scores_place_only.npz',
          bump_score_orginal_only=bump_score_orginal_only,
          bump_score_others_only=bump_score_others_only,
          bump_score_diff_only=bump_score_diff_only,
          bump_score_orginal_std_only=bump_score_orginal_std_only,
          bump_score_others_std_only=bump_score_others_std_only,
          bump_score_diff_std_only=bump_score_diff_std_only)

  # plt.show()

  end_time = time.time()

  # 计算并打印执行时间
  execution_time = end_time - start_time
  print(f"Total execution time: {execution_time} seconds")