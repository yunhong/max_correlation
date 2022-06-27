# given a set of records (x, y), find w to maximize cor(x * w, y)
# xi is a vector, yi is a constant
# X is a list, Xi is a vector with dimension d
# given X, Y, W, calculate the correlation
def dot_product(V1, V2, n):
    v = 0
    for i in range(n):
        #if(n > len(V1)) or (n > len(V2)): print('debug:\t' + str([V1, V2, n]))
        v += V1[i] * V2[i]
    return v

def multiply_vector(M, V): # matrix M times vector V 
    n = len(M)
    m = len(V)

    if(len(M[0]) != m): 
        print('error, length not equal: ' + str([n, len(M[0]), len(V)]))
        return []

    Z = n * [0]
    for i in range(n): 
        Z[i] = dot_product(M[i], V, m)
    # end for i
    return Z

# calculate correlation between X*W and Y
def cal_cor_linear(X, Y, W):
    Z = multiply_vector(X, W)
    return cal_cor(Z, Y)

EPS = 0.000001
import math
def cal_cor(Z, Y):
    n = len(Z)

    avg_yz = dot_product(Z, Y, n) / n

    uZ = float(sum(Z)) / n
    avg_z2 = dot_product(Z, Z, n) / n
    std_z = math.sqrt(avg_z2 - uZ * uZ)

    uY = float(sum(Y)) / n
    avg_y2 = dot_product(Y, Y, n) / n
    std_y = math.sqrt(avg_y2 - uY * uY)

    if(std_y * std_z <= EPS): 
        return None
    else:
        return (avg_yz - uY * uZ) / (std_y * std_z)

# M is the input matrix, Y labels, W are model parameters 
# get constants a, b, c
def get_coefficients(X, Y):
    #print('debug 54' + str([X, Y]))
    m = len(X[0])
    a = m * [0]
    b = m * [0]
    c = m * [0]
    for k in range(m):
        c[k] = m * [0]

    n = len(X)
    n_float = float(n)
    for j in range(m):
        v = 0
        for i in range(n): v += X[i][j]

        b[j] = v/n_float
    # end for b
    
    for j in range(m):
        for k in range(j, m):
            v = 0
            for i in range(n):
                v += X[i][j] * X[i][k]
            c[j][k] = v / n_float

            if(j < k): c[k][j] = c[j][k]
    # end for c

    uY = sum(Y) / n_float
    for j in range(m):
        v = 0
        for i in range(n): 
            v += X[i][j] * (Y[i] - uY)

        a[j] = v / n_float
    # end for
    
    return [a, b, c]

def str_k(v, k):
    s = int(v * pow(10, k)) * 1.0 / pow(10,k)
    return str(s)

EPS = 0.000001
def stop_condition(W, dW, stepsize, X, Y, P, Q, obj_pre, cnt):
    # calculate objective function, correlation during the current step
    cor = cal_cor_linear(X, Y, W)
    
    obj = P / Q
    #print('iter ' + str(cnt) + ', Correlation of X*W and Y: ' + str_k(cor, 4) + ',\tObjective: ' + str_k(obj, 4))

    if(obj_pre != None) and (obj - obj_pre < EPS):
        return True
    abs_dW = 0
    for j in range(len(W)):
        abs_dW = max(abs_dW, abs(dW[j]))
    abs_dW *= stepsize 
    if(abs_dW < EPS): return True
    return False

def normalize(W, index):
    w = W[index]
    for i in range(len(W)):
        W[i] /= w
    
# solve it through Gradient Descent, with default lambda_l2 = 0
def solve_GraDes(X, Y, columns, lambda_l2=0.0):
    [a, b, c] = get_coefficients(X, Y)
    m = len(X[0])

    if(len(columns) == 13): pos_LDT = 5
    else: pos_LDT = 3
    flag_iter_norm = True # normalize the weight of LDT to be 1 during each iteration

    import random
    W = []
    for i in range(m):
        v = random.random()
        W.append(v)
    # end 
    normalize(W, pos_LDT)

    W_init = list(W)

    stepsize = 0.0001

    obj_pre = None
    cnt = 0
    while(True):
        cnt += 1
        [dW, P, Q] = get_gradient(W, a, b, c, lambda_l2)
        obj = P / Q
        if(stop_condition(W, dW, stepsize, X, Y, P, Q, obj_pre, cnt)): break
        obj_pre = obj
        for j in range(m):
            W[j] += stepsize * dW[j] # Obj = value - lambda w^2, max obj -> +dw 
        if(flag_iter_norm): normalize(W, pos_LDT)

        if(cnt % 100000 == 0): 
            print('cnt\t' + str(cnt))
            for j in range(m):
                print(str_k(W[j], 4) + '\t' + columns[j])
    # end 

    print('converge rounds: ' + str(cnt))
    #for j in range(m): print(str_k(W[j], 4) + '\t' + columns[j])

    normalize(W, pos_LDT)

    return [W, W_init]

# default is lambda_l2 = 0, no regularizaiton, otherwise L2 regularization
def get_gradient(W, a, b, c, lambda_l2=0):
    # O(w) = P(w) / Q(w), calculate P and Q first
    m = len(W)

    aw_dot_prod = dot_product(a, W, m)
    P = aw_dot_prod**2

    bw_dot_prod = dot_product(b, W, m)
    cww_dot_prod = 0
    for j in range(m):
        for k in range(m):
            cww_dot_prod += W[j] * W[k] * c[j][k]
    # end for j,k

    Q = cww_dot_prod - bw_dot_prod**2

    dW = m * [0]
    for j in range(m):
        vj = dot_product(W, c[j], m) - b[j] * bw_dot_prod
        dW[j] = aw_dot_prod * a[j] * Q - P * vj
    # end for j

    Q_constant = 2.0 / Q**2
    for j in range(m):
        dW[j] *= Q_constant 

    # add regularization terms here, maximize the objective function, thus minus regularization term
    for j in range(m):
        dW[j] -= 2.0 * lambda_l2 * W[j] 

    return [dW, P, Q]


def load_data(infile):
    X = []
    Y = []

    fp = open(infile)
    for line in fp:
        tokens = map(int, line.strip().split('\t'))
        label = tokens[0]
        vector = tokens[1:]
        X.append(vector)
        Y.append(label)
    # end for
    fp.close()
    return [X, Y]


columns = ['qs_reformulation', 'qs_abandonment', 'qs_quickretry', 'ad_click', 
           'dd_ldt_clk', 'alg_ldt_clk', 'ad_ldt_clk', 
           'dd_norm_clk', 'alg_norm_clk', 'ad_norm_clk', 
           'dd_sdt_clk', 'alg_sdt_clk', 'ad_sdt_clk']

def customize(X):
    
    # remove ad_click, combine clicks of different types
    columns_new = ['qs_reformulation', 'qs_abandonment', 'qs_quickretry', 
           'ldt_clk', 'norm_clk', 'sdt_clk']

    X_new = []
    for ux in X:
        ux_new = list(ux[0:3])

        ldt_clk = ux[4] + ux[5] + ux[6]
        norm_clk = ux[7] + ux[8] + ux[9]
        sdt_clk = ux[10] + ux[11] + ux[12]

        ux_new += [ldt_clk, norm_clk, sdt_clk]
        X_new.append(ux_new)

    # end for
    return [X_new, columns_new]

# solve it with L2 regularizaiton, default lambda_l2 = 0
def process(X, Y, columns, lambda_l2=0):
    W, W_init = solve_GraDes(X, Y, columns, lambda_l2)

    cor = cal_cor_linear(X, Y, W)
    print('correlation of X*W and Y in training:\t' + str_k(cor, 4))

    for i in range(len(W)):
        print(str_k(W_init[i], 2) + '->\t' + str_k(W[i], 2) + '\t' + columns[i])

    return W


def one_run(X_new, Y, columns_new, X_other_new, Y_other, lambda_l2):
    W_new = process(X_new, Y, columns_new, lambda_l2)

    X_test, Y_test = load_data(infile_test)
    X_test_new, columns_new = customize(X_test)

    cor_test = cal_cor_linear(X_test_new, Y_test, W_new)
    print('correlation test: ' + str_k(cor_test, 4)) 

    X_trntst_new = X_new + X_test_new
    Y_trntst = Y + Y_test
    cor_trntst = cal_cor_linear(X_trntst_new, Y_trntst, W_new)
    print('correlation train/test: ' + str_k(cor_trntst, 4)) 

    cor_other = cal_cor_linear(X_other_new, Y_other, W_new)
    print('correlation other: ' + str_k(cor_other, 4))
 
    return [W_new, cor_test, cor_trntst, cor_other]

    

import sys
if(len(sys.argv) == 1) or (sys.argv[1] == '-h'): 
    print('input file_label_ux_components, output maximal correlation between label and ux_score\n')
    exit()

infile_train = sys.argv[1]
infile_test = sys.argv[2]

if(len(sys.argv) == 3): 
    lambda_l2 = 0.0 # default lambda_l2 is zero
else: 
    lambda_l2 = float(sys.argv[3])

print('correlation max: infile_train/tst ' + infile_train + ', ' + infile_test + ', lambda_l2 ' + str(lambda_l2))

X, Y = load_data(infile_train)
X_new, columns_new = customize(X)

#W = process(X, Y, columns, lambda_l2)

print('lambda_l2: ' + str(lambda_l2))

infile_other = '/Users/yunhongz/onedrive/code/utility_federation/correlation/QS_data//label_ux_pc300'
X_other, Y_other = load_data(infile_other)
X_other_new, columns_new = customize(X_other)


cor_trntst_max = 0
W_new_max = None
info_max = None
for t in range(50):
    [W_new, cor_test, cor_trntst, cor_other] = one_run(X_new, Y, columns_new, X_other_new, Y_other, lambda_l2)
    if(cor_trntst > cor_trntst_max):
        info_max = [W_new, cor_test, cor_trntst, cor_other] 
        cor_trntst_max = cor_trntst
        W_new = info_max[0]
        print('max cor_trntst\t' + str(cor_trntst_max))
        for i in range(len(W_new)):
            print(str_k(W_new[i], 2) + '\t' + columns_new[i])
# end for

[W_new, cor_test, cor_trntst, cor_other] = info_max
print('max cor_trntst\t' + str(cor_trntst))
print('max cor_other\t' + str(cor_other))
W_new = info_max[0]
for i in range(len(W_new)):
    print(str_k(W_new[i], 2) + '\t' + columns_new[i])

