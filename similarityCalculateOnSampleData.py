import gensim
import pickle
import re
import string
import math
import random
import numpy as np
import tensorflow as tf

model = gensim.models.KeyedVectors.load_word2vec_format("/home/haodong/shen_cnn/word/GoogleNews-vectors-negative300.bin", binary=True)
input = []
test_data =[]

batch_size = 64#extend to 128
learning_rate = 0.001

def get_input_data():
    with open('data/yahoo_clean_reduce_train', 'rb') as file:
    #with open('data/output_pickle1.pickle','rb') as file:
        while True:
            try:
                que_and_ans = []
                input_dict = pickle.load(file)
                #print(len(input_dict))
                key = input_dict.keys()[0]
                que_and_ans.append(key.split())
                answers_in_dict = input_dict[key]
                ans = []
                for i in range(len(answers_in_dict)):
                    ans.append(answers_in_dict[i].split())
                que_and_ans.append(ans)
                input.append(que_and_ans)
            except:
                print("catch pickle file end exception")
                break
def get_test_data():
    with open('data/yahoo_clean_reduce_test','rb') as file:
        while True:
            try:
                que_and_ans = []
                input_dict = pickle.load(file)
                #print(len(input_dict))
                key = input_dict.keys()[0]
                que_and_ans.append(key.split())
                answers_in_dict = input_dict[key]
                ans = []
                for i in range(len(answers_in_dict)):
                    ans.append(answers_in_dict[i].split())
                que_and_ans.append(ans)
                test_data.append(que_and_ans)
            except:
                print("catch pickle file end exception")
                break
def get_small_input_data():
    with open('data/small_yahoo_clean_reduce_train','rb') as file:
        input_dict = pickle.load(file)
        for key in input_dict.keys():
            que_and_ans = []
            que_and_ans.append(key.split())
            answers_in_dict = input_dict[key]
            ans = []
            for i in range(len(answers_in_dict)):
                ans.append(answers_in_dict[i].split())
            que_and_ans.append(ans)
            input.append(que_and_ans)
def get_batch(batch_size,input):
    total_data_num = len(input)
    start_index = 0
    for index in range(1,total_data_num/batch_size+1):
        end_index = start_index + batch_size
        similarity_matrix_batch = []
        ys = []
        for k in range(start_index,end_index):
            que_best_ans_similar = []
            que_ans_similar = []
            question = input[k][0]
            if len(question) == 0:
                continue
            best_answer = input[k][1][0]
            if (len(best_answer)>0):
                for i in range(30):
                    similar_row = []
                    for j in range(50):
                        try:
                            similar_row.append(math.fabs(model.similarity(question[i%len(question)],
                                                                         best_answer[j%len(best_answer)])))
                        except KeyError:
                            similar_row.append(0.0)
                    que_best_ans_similar.append(similar_row)
                similarity_matrix_batch.append(que_best_ans_similar)
                ys.append([1,0])
            random_answer_index = random.randint(0,total_data_num-1)
            while random_answer_index == k or len(input[random_answer_index][1]) == 0\
                    or len(input[random_answer_index][1][0]) ==0 :
                random_answer_index = random.randint(0,total_data_num)
            random_ans = input[random_answer_index][1][0]
            for i in range(30):
                similar_row = []
                for j in range(50):
                    try:
                        similar_row.append(math.fabs(model.similarity(question[i%len(question)],
                                                                    random_ans[j%len(random_ans)])))
                    except KeyError:
                        similar_row.append(0.0)
                que_ans_similar.append(similar_row)
            similarity_matrix_batch.append(que_ans_similar)
            ys.append([0,1])
        start_index += batch_size
        yield similarity_matrix_batch,ys

def get_test_batch(test_data):
    test_data_num = len(test_data)
    for j in range(0, test_data_num):
        question = test_data[j][0]
        best_answer = test_data[j][1][0]
        if len(question) == 0:
            continue
        if len(best_answer) == 0:
            continue
        similarity_matrix = []
        que_and_ans_similar = []
        for que_i in range(30):
            similar_row = []
            for ans_i in range(50):
                try:
                    similar_row.append(math.fabs(model.similarity(question[que_i % len(question)],
                                                                  best_answer[ans_i % len(best_answer)])))
                except KeyError:
                    similar_row.append(0.0)
            que_and_ans_similar.append(similar_row)
        similarity_matrix.append(que_and_ans_similar)
        random_ans_i = 1
        count = 1
        while count <= 5:
            random_ans_index = j + random_ans_i
            if random_ans_index >= test_data_num:
                random_ans_index = j - random_ans_i
            random_ans = test_data[random_ans_index][1][0]
            if len(random_ans) == 0:
                random_ans_i += 1
                continue
            que_and_ans_similar = []
            for que_i in range(30):
                similar_row = []
                for ans_i in range(50):
                    try:
                        similar_row.append(math.fabs(model.similarity(question[que_i % len(question)],
                                                                      random_ans[ans_i % len(random_ans)])))
                    except KeyError:
                        similar_row.append(0.0)
                que_and_ans_similar.append(similar_row)
            similarity_matrix.append(que_and_ans_similar)
            random_ans_i += 1
            count += 1
        yield similarity_matrix

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def cnn(batch_size,learning_rate,input,test_data):
    print("start")
    xs = tf.placeholder(tf.float32, [None, 30,50])
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_drop = tf.placeholder(tf.float32)
    x_input = tf.reshape(xs,[-1,30,50,1])
    #C1
    print('C1')
    W_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
    #M1
    print('M1')
    h_pool1 = max_pool_2x2(h_conv1)
    #C2
    print('C2')
    W_conv2 = weight_variable([5, 5, 20, 50])
    b_conv2 = bias_variable([50])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #M2
    print('M2')
    h_pool2 = max_pool_2x2(h_conv2)
    #print('h_pool2 shape',h_pool2.shape)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*13*50])
    W_fc1 = weight_variable([8*13*50, 500])
    b_fc1 = bias_variable([500])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)

    W_fc2 = weight_variable([500, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),
                       reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    print('session')
    sess = tf.Session()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    print('init')
    sess.run(init)
    for i,(batch_xs,batch_ys) in enumerate(get_batch(batch_size,input)):
        sess.run(train_step, feed_dict={xs: np.array(batch_xs), ys: np.array(batch_ys), keep_drop: 0.5})
        loss = sess.run(cross_entropy, feed_dict={xs: np.array(batch_xs), ys: np.array(batch_ys), keep_drop:0.8})
        print('batch {} loss {}'.format(i,loss))

        #accuracy
        y_pre = sess.run(prediction, feed_dict={xs: np.array(batch_xs), keep_drop: 1})
        #print(y_pre)
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(np.array(batch_ys), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: np.array(batch_xs), ys: np.array(batch_ys), keep_drop: 1})
        print('batch {} accuracy {}'.format(i,result))

        #nDCG6
        if True:
            nDCG6 = 0.0
            for j,test_xs in enumerate(get_test_batch(test_data)):
                if (len(test_xs)!=6):
                    print('{}th is {}'.format(j,len(test_xs)))
                pre = sess.run(prediction,feed_dict={xs:np.array(test_xs),keep_drop:1})
                if (len(pre)!=6):
                    print('{}th pre is {}'.format(j,len(pre)))
                c = 0
                for ans_i in range(1,6):
                    if pre[0][1] <= pre[ans_i][1]:
                        c += 1
                if c == 0:
                    nDCG6 += 1
                else:
                    nDCG6 += 1/np.log2(c+1)
            print(nDCG6/(j+1))


if __name__ == '__main__':
    get_input_data()
    get_test_data()
    print(len(input))
    # for i,(similartity_matrix,ys) in enumerate(get_batch(batch_size)):
    #     print(len(similartity_matrix))
    #     similartity_matrix_array = np.array(similartity_matrix)
    #     print(similartity_matrix_array.shape)
    # print('cnn_start')
    # for i,similarity_matrix in enumerate(get_test_batch(test_data)):
    #     if len(similarity_matrix) != 6:
    #         print ('{}th is {}'.format(i,len(similarity_matrix)))
    cnn(batch_size,learning_rate,input,test_data)