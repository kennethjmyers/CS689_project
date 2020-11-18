from copy import deepcopy
import numpy as np

##### HOW TO GENERATE LOGICAL REPRESENTATIONS

### Leaf encoding
# run a DFS on the tree, store nodes in a stack

# when you reach a leaf, write the logical statements for 
# pi(l) which is stack[1:] n wl where wl is the weight value of the leaf

# DFS
def dfs(tree, tree_num=0, X_input=None, epsilon=160, scale=0.1, verbose=False):
    X_input = np.array(X_input)  # if dataframe this converts to numpy
    paths = []
    pi_l = []  # this will be a list of lists of every pi_l ie: ['xi_1 <= 1000', 'xi_2 > 6000'..., wl]
    stack=[0]  # initialize stack for tree
    pi_l_stack=[]  # a stack for holding pi_l definitions based on position in tree
    n_nodes = tree.node_count
    threshold = tree.threshold 
    feature = tree.feature
    left_searched = [False for _ in range(n_nodes)]
    right_searched = [False for _ in range(n_nodes)]
    children_left = tree.children_left
    children_right = tree.children_right
    values = tree.value  # I think these are the average value (or something) of a node based on training data
    
    reals = set(['out',])  # this is a set of reals that we will be adding to to instantiate later, any time we encounter a variable we will add to it
    variables = set()  # this is used for variable constraints
    
    while True:
        this_node = stack[-1]
        this_threshold = threshold[this_node]
        this_feature = feature[this_node]

        searched_left_already = left_searched[this_node]
        searched_right_already = right_searched[this_node]
        
        # define tree pruning
        if children_left[this_node] == -1:  # if leaf node we dont do pruning even if a threshold is defined
            pass  # no pruning
        elif X_input is not None:
            lower_bound = X_input[this_feature]-epsilon
            upper_bound = X_input[this_feature]+epsilon
            if lower_bound <= this_threshold <= upper_bound:
                pass  # no pruning because threshold within boundary, permissiable values on either side of threshold
            elif this_threshold < lower_bound:
                left_searched[this_node]=True  # prune left side because no possible value for this allowable input range will go down left side
            elif upper_bound < this_threshold:
                right_searched[this_node]=True # prune right side because no possible value for this allowable input range will go down right side
        
        # check if we need to search any more nodes at this level or go up previous one
        if searched_left_already and searched_right_already and this_node==0:
            break  # special condition from pruning where we have returned to the root of the tree after searching everything else
        elif searched_left_already and searched_right_already:
            stack.pop(-1)  # pop the last node off of the stack so we go one level up on the next run
            pi_l_stack.pop(-1)
            continue
        
        if not searched_left_already:  # search the left_child
            child = children_left[this_node]
            this_op = f'x_{this_feature} <= {this_threshold}'  # I wasn't sure exactly how these should be defined
            left_searched[this_node]=True
        else:  # search the right child
            child = children_right[this_node]
            this_op = f'x_{this_feature} > {this_threshold}'
            right_searched[this_node]=True

        if child==-1:  # at leaf
            # append path to this node + the value of this node
            this_path = deepcopy(stack[1:])+[deepcopy(values[this_node][0,0])]
            this_pi_l = deepcopy(pi_l_stack[:])+[f'wl_{tree_num}=='+deepcopy(str(scale*values[this_node][0,0]))]
            reals.add(f'wl_{tree_num}')  # adding this leaf node to the list of reals we need to instantiate
            paths.append(this_path)
            pi_l.append(this_pi_l)
            result='And('+', '.join(this_pi_l)+')'
            if verbose:
                print(this_path, result)

            if this_node == n_nodes-1:
                break  # we have searched all paths and can break
            stack.pop(-1)  # pop the last node off of the stack so we go one level up on the next run
            pi_l_stack.pop(-1)
        else:  # not at leaf
            # append the child to the stack and continue search
            stack.append(child)     
            pi_l_stack.append(this_op)
            reals.add(f'x_{this_feature}')  # add this feature to the set of reals, its okay if we choose to discard this path later
            variables.add(f'x_{this_feature}')
            
    return pi_l, reals, variables

# pruning: using an input x_i', compare the difference at every node in the DFS x_i
# if |x_i-x_i'| < epsilon where epsilon is the predefined robustness factor
# then keep node, otherwise pop current node from the stack and 
# stop searching this branch of the tree (return to previous level)

### Tree encoding
# then represent PI(D) as V(pi(l)) a disjunction of all pi(l) found during DFS

### GBM encoding
# lastly encode the full model instelf, (n(PI(D_i))) n (out=sum(wl_i))
# I haven't fulling figured out how to represent the out=sum(wl_i) part yet.
# wl_i should be the one value returned from each tree PI(D_i), we might just 
# need the additional predicate D_i(x)=wl_i meaning that putting x into the decision tree returns wl_i
# but I'm not totally sure how this is expressed for Z3

# gamma_R is the model encoding and referenced in the paper
def get_gamma_R(gbr_model, X_input, epsilon=160, delta=100000):
    all_pi_l = []
    all_reals = set()
    all_variables = set()
    scale = gbr_model.learning_rate
    for t in range(len(gbr_model.estimators_)):
        tree = gbr_model.estimators_[t,0].tree_
        this_pi_l, reals, variables = dfs(tree, tree_num=t, X_input=X_input, epsilon=epsilon, scale=scale)
        all_pi_l.append(this_pi_l)
        all_reals = all_reals.union(reals)
        all_variables = all_variables.union(variables)
        
    all_PI_Ds = []
    for r in range(len(all_pi_l)):
        this_PI_D = all_pi_l[r]
        PI_D = '\tOr(\n\t\t'+',\n\t\t'.join(['And('+', '.join(this_pi_l)+')' for this_pi_l in this_PI_D])+'\n\t)'
        all_PI_Ds.append(PI_D)

        # print('\nTree Representation PI(D):\n',PI_D,sep='')

    # construct variable constraints
    ### Robust property encoding - see section 5 of the paper this is easier than the previous stuff but it 
    variable_constraints = []
    for x_i in list(all_variables):
        this_index = int(x_i[2:])  # everything looks like x_i so indexing from x_[i..]
        i_value = X_input[this_index]
        var_robustness1 = f'{x_i} <= {i_value+epsilon}'
        var_robustness2 = f'{i_value-epsilon} <= {x_i}'  # ensures that x_i is withing these boundaries
        variable_constraints.append(f'And({var_robustness1}, {var_robustness2})')
        
    # out='wl_1+wl_2+...'
    out = 'out=='+'+'.join([f'wl_{t}' for t in range(len(all_pi_l))])
    
    # calculate the residual value for the input
    ### Robust property encoding - see section 5 of the paper this is easier than the previous stuff but it 
    X_sample = np.array(X_input).reshape(1,-1)
    input_regression_value = 0 # This is R(x), the residual value of the original input
    n_estimators = gbr_model.n_estimators
    scale = gbr_model.learning_rate
    for t in range(n_estimators):
        estimator = gbr_model.estimators_[t,0]  # this is a regressor so there is only one output hence the 0
        value_in_decision_tree = estimator.predict(X_sample)[0]
        input_regression_value += scale*value_in_decision_tree  
    
    output_robustness1 = f'{input_regression_value+delta} < out'
    output_robustness2 = f'out < {input_regression_value-delta}'  # these two lines will ensure that all out values are OUTSIDE delta of the input regression value, so that if there is a counter-example then it will be found
    output_robustness = f'Or({output_robustness2}, {output_robustness1})'
    
    gamma_R = ',\n'.join(all_PI_Ds)
    gamma_R = 'And(\n\t'+',\n\t'.join(variable_constraints)+',\n\t'+output_robustness+',\n\t'+out+',\n'+gamma_R+'\n)'

    # print(out)
    # print(gamma_R)
    return gamma_R, all_reals





