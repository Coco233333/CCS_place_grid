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
import random

# 设置工作路径
work_dir = f'/home/yulingwu/CCS_place_grid/Storing_multiple_maps/test/result_20250709/W0'
os.makedirs(work_dir, exist_ok=True)

place_num = 800
grid_num = 20
module_num = 10

# Ws = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5] # 0.01, 0.02, 
# Ws = [0]

Ws = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]  # 

for emer, W in enumerate(Ws):
  # 设置次级目录
  sub_dir = os.path.join(work_dir, f'{W}')
  os.makedirs(sub_dir, exist_ok=True)
  os.chdir(sub_dir)

  # 记录开始时间
  start_time = time.time()

  # Define constants
  z_min, z_max = 0, 20
  a_p = 0.5
  Spacing = bm.linspace(6, 20, module_num)
  simulaiton_num = 30

  # Preallocate arrays
  capacity = bm.Variable(bm.zeros((simulaiton_num)))

  for k in range(simulaiton_num):
    # 设置随机种子
    seed = k
    np.random.seed(seed)
    bm.random.seed(seed)
    random.seed(seed)

    Place_cell = Place_net(z_min=z_min, z_max=z_max, map_num=50, neuron_num=place_num, place_num=place_num,
                          noise_stre=0.5)
    maps = Place_cell.map
    place_index = Place_cell.place_index

    bump_score_diff =  1
    map_num = 2

    # Main loop
    while bump_score_diff >= 0.2:
      
      Place_cell.conn_mat = bm.zeros([Place_cell.neuron_num, Place_cell.neuron_num])
      for i in range(map_num):
        conn_mat = Place_cell.make_conn(Place_cell.map[i])
        if i >= 1:
          mean_conn = bm.mean(conn_mat)
          conn_mat = conn_mat - mean_conn
          # print(bm.sum(conn_mat))
        Place_cell.conn_mat[bm.ix_(Place_cell.place_index[i], Place_cell.place_index[i])] += conn_mat
      bm.fill_diagonal(Place_cell.conn_mat, 0)

      # Grid module list initialization
      Gird_module_list = [Grid_net(L=Spacing[module], maps=maps[:i], place_index=place_index[:i], neuron_num=grid_num, J0=5, W0=W,
                                  a_g=a_p / Spacing[module] * 2 * bm.pi) for module in range(module_num)]
      map_index = 0

      def run_net(indices, loc, input_stre):
        r_hpc = Place_cell.r
        output = bm.zeros(place_num, )
        for Grid_cell in Gird_module_list:
          Grid_cell.step_run(indices, r_hpc=r_hpc, loc=loc, input_stre=input_stre, map_index=map_index)
          output += Grid_cell.output
        Place_cell.step_run(indices, loc=loc, map_index=map_index, input_stre=input_stre, input_g=output)
        return Place_cell.r.value


      total_time = 5000
      start_time = 1000
      indices = bm.arange(total_time)
      loc = bm.zeros(total_time) + (z_max + z_min) / 2
      input_stre = bm.zeros(total_time)
      input_stre[:start_time] = 10.

      us = bm.for_loop(run_net, (indices, loc, input_stre), progress_bar=False)
      u = us[-1]
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
      

      bump_score = compute_bump_score(u, loc_candidate, map_num=map_num, Place_cell=Place_cell, place_index=place_index[:i]) # coupled net  
      bump_score_diff = bump_score[0] - bm.max(bump_score[1:])
      map_num += 1
    
    capacity = capacity.at[k].set(map_num - 2)  # Store the final map_num
    print(f'W0={W}, simulation {k+1}/{simulaiton_num}, capacity={map_num - 2}')

  # Save the data
  np.savez('capacity.npz',
          capacity=capacity)