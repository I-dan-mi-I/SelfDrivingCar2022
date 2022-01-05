# Executable file (main file of project)
# my code doesn't match PEP-8, but I don't care :)
# tensorflow throws in a lot of errors, but I staggered tf v 2
# to fix this, use tf v 1.13

# Cheers for
# https://github.com/MorvanZhou for source code of SumTree
# https://github.com/jaromiru for source code of Memory
# https://github.com/awjuliani for source code of def update_target_graph()

# для создания окружения физической модели
from pyglet.gl import *
from Game import Game

# для работы с нейросетями
import tensorflow as tf
import numpy as np
import random

# для работы с файлами
import os

# инициализация физической модели и списка разрешённых действий
game = Game()
possible_actions = np.identity(game.no_of_actions, dtype=int).tolist()

# параметры физической модели
displayWidth = 1800
displayHeight = 1000
FPS = 30.0

# параметры модели
state_size = [game.state_size]
action_size = game.no_of_actions
learning_rate = 0.00025

# параметры обучения
total_episodes = 50000
max_steps = 5000
batch_size = 64

# параметры эпизодов
load = True
starting_episode = 0
episode_render = True

# параметры Q-обучеия
max_tau = 10000
gamma = 0.95

# параметры explore
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005

# параметры памяти
memory_size = 100000
pretrain_length = memory_size

# параметры начала обучения и загрузки обученных агентов
training = False
load_traing_model = False
load_training_model_number = 300


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # Здесь мы используем tf.variable_scope, чтобы знать, какую сеть мы используем (DQN или target_net)
        # это будет полезно, когда мы будем обновлять наши w-параметры (копируя параметры DQN)
        with tf.variable_scope(self.name):
            # создаём placeholder (check the Tensorflow manual to know how this work)
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # target_Q = R(s,a) + y_max * Q_hat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.dense1 = tf.layers.dense(inputs=self.inputs_,
                                          units=256,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="dense1")

            self.dense2 = tf.layers.dense(inputs=self.dense1,
                                          units=256,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="dense2")

            # Здесь мы разделяем на два потока

            # расчёт V(s)
            self.value_fc = tf.layers.dense(inputs=self.dense2,
                                            units=256,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # расчёт A(s, a)
            self.advantage_fc = tf.layers.dense(inputs=self.dense2,
                                                units=256,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # агрегация слоёв
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # прогноз Q
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


# инициализация сети
tf.reset_default_graph()
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)

        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data

        self.update(tree_index, priority)

        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):

    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1

    def __init__(self, capacity):

        self.tree = SumTree(capacity)

    def store(self, experience):

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority

        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):

            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority

            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            if b_ISWeights[i, 0] == 0:
                print(n, sampling_probabilities, self.PER_b, max_weight)
            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# инициализаиция памяти
memory = Memory(memory_size)

# рендеринг окруженеия
game.new_episode()

# предварительное обучение
if training:
    for i in range(pretrain_length):
        # если первая модель
        if i == 0:
            state = game.get_state()

        # случайное действие
        action = random.choice(possible_actions)

        # награда (недооценивайте её важность, она будто зарплата для программиста)
        # (μ_μ) у меня даже этого нет
        # ботов для диса писать веселее
        reward = game.make_action(action)

        # эпизод закончился?
        done = game.is_episode_finished()

        # мамкин водятел въехал в стену
        if done:
            next_state = np.zeros(state.shape)

            experience = state, action, reward, next_state, done
            memory.store(experience)

            game.new_episode()

            state = game.get_state()

        # не въехал в стену
        else:
            next_state = game.get_state()

            experience = state, action, reward, next_state, done
            memory.store(experience)

            state = next_state


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # необращайте внимания на неиспользумый actions
    # так надо
    # я не программист, мне можно оставлять такие куски, ибо мне в лом чинить этот момент

    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)

    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability


def update_target_graph():
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


saver = tf.train.Saver()


class GameWindow(pyglet.window.Window):
    """ можно я хоть в коде буду говорить, что это игра?
    хочется умереть, у всех праздники, а я тут дописываю
    ψ(▼へ▼メ)～→
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        # фон
        backgroundColor = [10, 0, 0, 255]
        backgroundColor = [i / 255 for i in backgroundColor]
        glClearColor(*backgroundColor)

        self.sess = tf.Session()
        game.new_episode()
        self.state = game.get_state()
        self.nextState = []
        self.loadSession()

    def loadSession(self):
        if load_traing_model:
            directory = "./allModels/model{}/models/model.ckpt".format(load_training_model_number)
            saver.restore(self.sess, directory)
        else:
            saver.restore(self.sess, "./models/model.ckpt")

    def on_draw(self):
        game.render()

    def update(self, dt):
        exp_exp_tradeoff = np.random.rand()

        if load_traing_model:
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * load_training_model_number* 100)
        else:
            explore_probability = 0.0001

        if explore_probability > exp_exp_tradeoff:
            action = random.choice(possible_actions)

        else:
            Qs = self.sess.run(DQNetwork.output,
                               feed_dict={DQNetwork.inputs_: self.state.reshape((1, *self.state.shape))})

            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]

        game.make_action(action)
        done = game.is_episode_finished()

        if done:
            game.new_episode()
            self.state = game.get_state()
        else:
            self.next_state = game.get_state()
            self.state = self.next_state


# надо сделать этот блок
# мой компьютер с gtx 1050 ti и фуфыксом взорвётся от этих нейросетей
print("training")
if training:
    with tf.Session() as sess:

        if load:
            saver.restore(sess, "./models/model.ckpt")
        else:
            sess.run(tf.global_variables_initializer())

        decay_step = 0

        tau = 0

        # инициализируем игру
        game.new_episode()

        # обновляем веса
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(starting_episode, total_episodes):
            step = 0

            episode_rewards = []

            game.new_episode()

            state = game.get_state()

            while step < max_steps:
                step += 1

                tau += 1

                decay_step += 1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                reward = game.make_action(action)

                done = game.is_episode_finished()

                episode_rewards.append(reward)
                # if step >= max_steps:
                #    print('Episode: {}'.format(episode),
                #          'Total reward: {}'.format(np.sum(episode_rewards)),
                #          'Explore P: {:.4f}'.format(explore_probability))
                

                if done:
                    next_state = np.zeros(state.shape, dtype=np.int)

                    step = max_steps

                    total_reward = np.sum(episode_rewards)

                    # вывод статистики для обучения, вдруг кому-то это нужно
                    # да есть tensorboard, но я ленивая зараза
                    # ( ﾟ，_ゝ｀)
                    # print('Episode: {}'.format(episode),
                    #      '\tTotal reward: {:.4f}'.format(total_reward),
                    #      # '\tTraining loss: {:.4f}'.format(loss),
                    #      '\tExplore P: {:.4f}'.format(explore_probability),
                    #      '\tScore: {}'.format(game.get_score()),
                    #      '\tlifespan: {}'.format(game.get_lifespan()),
                    #      '\tactions per reward gate: {:.4f}'.format(game.get_lifespan() / (max(1, game.get_score()))))

                    # пожалуй нужно сохранить это
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    next_state = game.get_state()

                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                    state = next_state

                # время кнута и пряника
                # случайное (?) кхм... объект из памяти
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states_mb = np.array([each[0][0] for each in batch], ndmin=2)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=2)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                # из DQNNetwork получим a'
                # из TargetNetwork посчитаем Q_val of Q(s',a')

                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

                # время магии
                # если эпизод заканчивается на s+1: Q_target = r
                # иначе Q_target = r + gamma * Q_target(s',a')

                # забавный факт для тех, кто всё таки будет читать код: сейчас 5 января 2022 года 12:22

                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # a'
                    action = np.argmax(q_next_state[i])

                    # если всё плохо (мы в терминальном состоянии), выдаём только вознаграждение
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb,
                                                               DQNetwork.ISWeights_: ISWeights_mb})

                # проверка и смена приоритетов
                # странно, почему я не сделал это в своей жизни?
                memory.batch_update(tree_idx, absolute_errors)

                if tau > max_tau:
                    # обновляем веса TargetNetwork из DQN
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            if (episode < 100 and episode % 5 == 0) or (episode % 1000 == 0):
                directory = "./allModels/model{}".format(episode)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = saver.save(sess, "./allModels/model{}/models/model.ckpt".format(episode))
                print("Model Saved")

            # каждые 5 эпизодов сохраняем модель
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")
else:
    window = GameWindow(displayWidth, displayHeight, "DanMi project 2022", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / FPS)
    pyglet.app.run()

print("testing")
with tf.Session() as sess:

    # подгружаем модель
    saver.restore(sess, "./models/model.ckpt")

    for i in range(10):
        game.new_episode()
        state = game.get_state()

        while not game.is_episode_finished():
            exp_exp_tradeoff = np.random.rand()

            explore_probability = 0.01

            if (explore_probability > exp_exp_tradeoff):
                action = random.choice(possible_actions)

            else:
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]

            game.make_action(action)
            window.draw(game)
            done = game.is_episode_finished()

            if done:
                break

            else:
                next_state = game.get_state()
                state = next_state
