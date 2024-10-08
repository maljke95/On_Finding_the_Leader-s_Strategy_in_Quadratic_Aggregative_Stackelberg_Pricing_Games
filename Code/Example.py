# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:14:21 2022

@author: marko
"""

import numpy as np
from scipy.linalg import block_diag
import cvxpy as cp
from quadprog import solve_qp
import os
from datetime import datetime 

def leder_obj(A_s, Ag, n, xi):
    
    return 0.5*xi @ A_s.T @ Ag @ A_s @ xi - xi @ A_s.T @ Ag @ n + 0.5*n @ Ag @ n

def prepare_follower_pseudo_grad(N, P, Q, S, r1, r2, r3, pi):
    
    for i in range(N):  
            for j in range(N):
                if i == j:
                    current = P   
                else:
                    current = Q   
                if j == 0:   
                    row = current 
                else:
                    row = np.concatenate((row, current),axis=1) 
            if i == 0:
                mat = row
            else: 
                mat = np.concatenate((mat, row), axis=0)
            
    vec = np.concatenate((np.concatenate((r1+S @ pi, r2+S@pi), axis=0), r3+S@pi),axis=0)
    
    F1 = mat
    F2 = vec
    
    return F1, F2

def calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, pi, x_init):
    
    F1, F2 = prepare_follower_pseudo_grad(N, P, Q, S, r1, r2, r3, pi)
    xi = x_init

    K_l = np.concatenate((np.concatenate((G, A), axis=0), -A), axis=0)
    k_r = np.concatenate((np.concatenate((h,b)), -b))
            
    qp_C = np.transpose(-K_l)
    qp_b = np.squeeze(-k_r)

    for iter in range(K):
        
        grad = F1 @ xi + F2
        
        x_hat = xi - gamma*grad
        
        qp_G = np.eye(len(x_hat))
        qp_a = x_hat
        
        try:
            
            sol,_,_,_,_,_ = solve_qp(qp_G, qp_a, qp_C, qp_b)
            xi = np.array(sol)
            
        except:
            print('fail')
            P2 = np.eye(len(x_hat))
            x = cp.Variable(len(P2))
            q = x_hat
            
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P2) - 2*q.T @ x),
                          [G @ x <= h,
                          A @ x == b])

            prob.solve()

            xi = np.squeeze(np.array(x.value))
    
    return xi

def active_constraints(xi, G, h):
    
    delta = G @ xi - h

    list_of_active_left = []
    list_of_active_right = []
    
    list_of_inactive_left = []
    list_of_inactive_right = []
    
    for i in range(len(delta)):
        
        if np.abs(delta[i])<0.001:
                
            list_of_active_left.append(np.squeeze(G[i,:]))
            list_of_active_right.append(h[i])
                
        else:
                
            list_of_inactive_left.append(np.squeeze(G[i,:]))
            list_of_inactive_right.append(h[i])
    
    G_A_comp = np.array(list_of_inactive_left)
    h_A_comp = np.array(list_of_inactive_right)
    
    G_A = np.array(list_of_active_left)
    h_A = np.array(list_of_active_right)
    
    return G_A_comp, h_A_comp, G_A, h_A

def adjust_constraints(G, h, A, b, x):
    
    G_A_comp, h_A_comp, G_A, h_A = active_constraints(x, G, h)
    
    A_new = A
    G_new = G_A_comp
    
    b_new = b
    h_new = h_A_comp
    
    if len(G_A)>0:
        
        A_new = np.concatenate((A_new, G_A), axis=0)
        b_new = np.concatenate((b_new, h_A), axis=0)
        
    if len(A_new)>len(x):
        
        A_new = A_new[:len(x), :]
        b_new = b_new[:len(x)]
        
    return A_new, b_new, G_new, h_new

def calculate_dual_var(P, q, G, h, A, b):
    
    x = cp.Variable(len(P))
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                     [G @ x <= h,
                      A @ x == b])
    
    prob.solve()
    xopt = np.squeeze(np.array(x.value))
    lamb = np.squeeze(np.array(prob.constraints[0].dual_value))
    
    return xopt, lamb       
        
def compute_jacobian(P, S, A_new, b_new, G_new, h_new, xi, lambi):
    
    K = np.concatenate((P, G_new.T), axis=1)
    K = np.concatenate((K, A_new.T), axis=1)
    
    K_row2 = np.concatenate((np.diag(lambi) @ G_new, np.diag(np.squeeze(G_new @ xi - h_new))), axis=1)
    K_row2 = np.concatenate((K_row2, np.zeros((len(G_new), len(A_new)))), axis=1)
    
    K_row3 = np.concatenate((A_new, np.zeros((len(A_new), len(A_new)+len(G_new)))), axis=1)
    
    K = np.concatenate((K, K_row2), axis=0)
    K = np.concatenate((K, K_row3), axis=0)
    
    Right = np.concatenate((-S, np.zeros((len(A_new)+len(G_new), S.shape[1]))), axis=0)
    dx_lam_nu = np.linalg.inv(K) @ Right
    
    jacobian = dx_lam_nu[:len(P),:]
    
    return jacobian    
    
def grid_search(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, x_init, N_des):

    p0 = np.linspace(0.0, 2.0, 20)
    p1 = np.linspace(0.0, 2.0, 20)
    p2 = np.linspace(0.0, 2.0, 20)
    
    list_cost = []
    list_p = []
    iter2 = 0
    
    A_s = 1.0*np.concatenate((np.concatenate((np.eye(3), np.eye(3)), axis=1), np.eye(3)),axis=1)
    
    Ag = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    
    for i in range(len(p0)):
        
        pi0 = p0[i]
        
        for j in range(len(p1)):
            
            pi1 = p1[j]
            
            for k in range(len(p2)):
                
                pi2 = p2[k]
                
                iter2 += 1
                print('Iteration: ', iter2)
                
                pi = np.array([pi0, pi1, pi2])
                
                xi = calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, pi, x_init)
                
                cost = leder_obj(A_s, Ag, N_des, xi)
                
                list_cost.append(cost)
                list_p.append(pi)   
                
    return list_cost, list_p    
    
if __name__ == '__main__':
    
    G1 = np.array([[-1.0, 0.0, 0.0], [0, -1.0, 0.0], [0, 0, -1.0], [0.0, 1.0, 0.0]])
    A1 = np.array([[1.0, 1.0, 1.0]])
    b1 = np.array([4.0])
    h1 = np.array([0.0, 0.0, 0.0, 1.5])
    
    G2 = np.array([[-1.0, 0.0, 0.0], [0, -1.0, 0.0], [0, 0, -1.0], [0.0, 0.0, 1.0]])
    A2 = np.array([[1.0, 1.0, 1.0]])
    b2 = np.array([5.0])
    h2 = np.array([0.0, 0.0, 0.0, 3.0])

    G3 = np.array([[-1.0, 0.0, 0.0], [0, -1.0, 0.0], [0, 0, -1.0], [1.0, 0.0, 0.0]])
    A3 = np.array([[1.0, 1.0, 1.0]])
    b3 = np.array([6.0])
    h3 = np.array([0.0, 0.0, 0.0, 2.0])    
   
    r1 = -np.array([3.0,2,5.3])
    r2 = -np.array([3.0,2,4])
    r3 = -np.array([4.0,1,5])
    
    S = np.array([[2.0,0,0],[0,4.0,0],[0,0,3.0]])
    
    coeff = 1.0
    
    S1 = S + coeff*np.diag(np.random.rand(3))
    S2 = S + coeff*np.diag(np.random.rand(3))
    S3 = S + coeff*np.diag(np.random.rand(3))

    G = np.array(block_diag(G1, G2, G3))
    h = np.concatenate((np.concatenate((h1, h2)), h3))
    
    A = np.array(block_diag(A1, A2, A3))
    b = np.concatenate((np.concatenate((b1, b2)), b3))
    
    N     = 3
    K_pi  = 0
    K     = 3000
    alpha = 0.01
    gamma = 0.001
    
    P  = np.array([[5.0,0,0],[0,5.0,0],[0,0,5.0]])
    Q  = 0.5*P
    
    A_s = 1.0*np.concatenate((np.concatenate((np.eye(3), np.eye(3)), axis=1), np.eye(3)),axis=1)    
    Ag = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    
    N_des = np.array([5.0, 3.0, 7.0])
    
    list_of_losses = []
    list_of_xi     = []
    list_of_pi     = []
    
    #----- Initial conditions -----
    
    pi = np.array([1.0,1.0,1.0])
    x0 = np.array(9*[1.0])/3
    
    cost = leder_obj(A_s, Ag, N_des, x0)
    list_of_losses.append(cost)
    list_of_xi.append(x0)
    
    #----- Prepare save -----
    
    current_folder = os.getcwd() + '/Results_dummy'
    
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    name = current_folder + "/" + date_time
    
    #----- Projected p -----
    
    projected = False
    
    p_max = 10.0
    p_min = 1.0
    
    G_pi = np.array([[1.0, 0, 0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    h_pi = np.array([p_max, -p_min, p_max, -p_min, p_max, -p_min])
           
    #----- Projected with armijo -----
    
    projected_with_armijo = True
    
    beta = 0.5
    s_line = alpha
    nu = 0.01
    
    list_of_counter_fail = []
    list_of_steps        = []
    
    #----- Prepare folder -----
    
    name = name + '_projected_' + str(projected)
    name = name + '_armijo_' + str(projected_with_armijo)

    if not os.path.isdir(name):
        os.makedirs(name) 
        
    for iteration in range(K_pi):
        
        print("It: ", iteration)
        
        list_of_pi.append(pi)
        
        if not projected_with_armijo or iteration == 0:
            xi = calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, pi, x0)
        
        list_of_xi.append(xi)
        cost = leder_obj(A_s, Ag, N_des, xi)
        list_of_losses.append(cost)    
    
        #----- Prepare best response -----
    
        q_br1 = r1 + S1 @ pi + Q @ (xi[3:6] + xi[6:])
        q_br2 = r2 + S2 @ pi + Q @ (xi[:3] + xi[6:])
        q_br3 = r3 + S3 @ pi + Q @ (xi[3:6] + xi[:3])    
        
        #---------------------------------
        
        A_new1, b_new1, G_new1, h_new1 = adjust_constraints(G1, h1, A1, b1, xi[:3])
        A_new2, b_new2, G_new2, h_new2 = adjust_constraints(G2, h2, A2, b2, xi[3:6])
        A_new3, b_new3, G_new3, h_new3 = adjust_constraints(G3, h3, A3, b3, xi[6:])
        
        xopt1, lamb1 = calculate_dual_var(P, q_br1, G_new1, h_new1, A_new1, b_new1)
        jacob1 = compute_jacobian(P, S, A_new1, b_new1, G_new1, h_new1, xi[:3], lamb1)
    
        xopt2, lamb2 = calculate_dual_var(P, q_br2, G_new2, h_new2, A_new2, b_new2)
        jacob2 = compute_jacobian(P, S, A_new2, b_new2, G_new2, h_new2, xi[3:6], lamb2)    
    
        xopt3, lamb3 = calculate_dual_var(P, q_br3, G_new3, h_new3, A_new3, b_new3)
        jacob3 = compute_jacobian(P, S, A_new3, b_new3, G_new3, h_new3, xi[6:], lamb3) 
        
        #----- Update the leader decision -----
        
        jacobian_full = np.concatenate((np.concatenate((jacob1.T, jacob2.T), axis=1), jacob3.T), axis=1)
        dJ_l_dx = A_s.T @ Ag @ A_s @ xi - A_s.T @ Ag @ N_des
        
        dJ_l_dpi = jacobian_full @ dJ_l_dx
        
        if projected:
            
            pi_hat = pi - alpha*dJ_l_dpi
            P_pi = np.eye(len(pi_hat))
            x_pi = cp.Variable(len(P_pi))
            q_pi = pi_hat
            
            prob = cp.Problem(cp.Minimize(cp.quad_form(x_pi, P_pi) - 2*q_pi.T @ x_pi),
                          [G_pi @ x_pi <= h_pi])

            prob.solve()

            pi = np.squeeze(np.array(x_pi.value))
            
        elif projected_with_armijo:
            
            found_step = False
            l = 0.0
            
            while not found_step:
                
                step = beta**l * s_line 
                
                pi_hat = pi - step*dJ_l_dpi
                P_pi = np.eye(len(pi_hat))
                x_pi = cp.Variable(len(P_pi))
                q_pi = pi_hat
            
                prob = cp.Problem(cp.Minimize(cp.quad_form(x_pi, P_pi) - 2*q_pi.T @ x_pi),
                          [G_pi @ x_pi <= h_pi])

                prob.solve()

                pi_plus = np.squeeze(np.array(x_pi.value)) 
                xi_plus = calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, pi_plus, x0)
                cost_plus = leder_obj(A_s, Ag, N_des, xi_plus)
                
                if cost - cost_plus >= nu * dJ_l_dpi @ (pi - pi_plus):
                    
                    found_step = True
                    
                else:
                    
                    l += 1
                    
            pi = pi_plus
            xi = xi_plus
            
            list_of_steps.append(step)
            list_of_counter_fail.append(l)
            
        else:
            
            pi = pi - alpha*dJ_l_dpi
                             
    np.save(name+'/list_of_losses.npy', np.array(list_of_losses))
    np.save(name+'/list_of_pi.npy'    , np.array(list_of_pi))
    np.save(name+'/list_of_xi.npy'    , np.array(list_of_xi))
    
    #----- Test the hypothesis -----
    
    p_init = np.array([1.87063, 2.87338, 1.0])
    
    k = 10.0
    
    p_edited = p_init + 1.0/np.diag(S)*k
    
    xi_1 = calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, p_init, x0)
    cost_1 = leder_obj(A_s, Ag, N_des, xi_1)
    
    xi_2 = calculate_NE(K, gamma, N, P, Q, S, r1, r2, r3, A, b, G, h, p_edited, x0)
    cost_2 = leder_obj(A_s, Ag, N_des, xi_2) 
    
    print(cost_1, cost_2)
 