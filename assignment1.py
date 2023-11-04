# %% [markdown]
# ### Read Data

# %%
# load data into dataset array
import gzip
from collections import defaultdict
import numpy as np
import tensorflow as tf

# %%
def readJSON(path):
    f = gzip.open(path, 'rt', encoding="utf-8")
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

dataset = []
for l in readJSON("train.json.gz"):
    dataset.append(l)

for user,game,review in dataset:
    review["played"] = 1

# %%
# train test split

from sklearn.model_selection import train_test_split

#train, valid = train_test_split(dataset, train_size=165000, random_state=0)
train = dataset[:165000]
valid = dataset[165000:]

# %%
# Get negative labels in vaidation
import random

def get_balanced_set(dataset, s):
    all_games = set()
    user_played = defaultdict(set)

    for user,game,review in dataset:
        all_games.add(review["gameID"])
        user_played[review["userID"]].add(review["gameID"])

    negative = []

    for user,game,review in s:
        not_played = all_games - user_played[user]
        new_game = random.choice(tuple(not_played))
        negative.append((user, new_game, {"played": 0}))

    return s + negative
    

# %% [markdown]
# ### Utility Functions

# %%
def writePredictions(infile, outfile, model):
    with open(outfile, 'w') as predictions:
        for l in open(infile):
            if l.startswith("userID"):
                predictions.write(l)
                continue
            u,g = l.strip().split(',')
            
            pred = model.predict(u,g)
            
            _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

        predictions.close()

# %%
class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb, itemIDs, userIDs):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))

# %% [markdown]
# ### Play Predictor

# %%
class PlayPredictor:

    def __init__(self):
        pass

    def fit(self, data, threshold=0.6, K=5, iters=100): # data is an array of (user, game, review) tuples
        self.topGames = self.getTopGames(threshold)

        self.userIDs = {}
        self.itemIDs = {}
        interactions = []

        for u,i,r in data:
            if not u in self.userIDs: self.userIDs[u] = len(self.userIDs)
            if not i in self.itemIDs: self.itemIDs[i] = len(self.itemIDs)
            interactions.append((u,i,r["played"]))
        
        items = list(self.itemIDs.keys())
        
        itemsPerUser = defaultdict(list)
        usersPerItem = defaultdict(list)
        for u,i,r in interactions:
            itemsPerUser[u].append(i)
            usersPerItem[i].append(u)

        def trainingStepBPR(model, interactions):
            Nsamples = 50000
            with tf.GradientTape() as tape:
                sampleU, sampleI, sampleJ = [], [], []
                for _ in range(Nsamples):
                    u,i,_ = random.choice(interactions) # positive sample
                    j = random.choice(items) # negative sample
                    while j in itemsPerUser[u]:
                        j = random.choice(items)
                    sampleU.append(self.userIDs[u])
                    sampleI.append(self.itemIDs[i])
                    sampleJ.append(self.itemIDs[j])

                loss = model(sampleU,sampleI,sampleJ)
                loss += model.reg()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients((grad, var) for
                                    (grad, var) in zip(gradients, model.trainable_variables)
                                    if grad is not None)
            return loss.numpy()
        
        optimizer = tf.keras.optimizers.Adam(0.1)
        self.modelBPR = BPRbatch(K, 0.00001, self.itemIDs, self.userIDs)

        for i in range(iters):
            obj = trainingStepBPR(self.modelBPR, interactions)
            if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
            
    def predict(self, user, game, threshold=0.5):
        if user in self.userIDs and game in self.itemIDs:
            pred = self.modelBPR.predict(self.userIDs[user], self.itemIDs[game]).numpy()
            return int(pred > threshold)
        else:
            return int(game in self.topGames)

    def getTopGames (self, threshold):
        gameCount = defaultdict(int)
        totalPlayed = 0

        for user,game,_ in readJSON("train.json.gz"):
            gameCount[game] += 1
            totalPlayed += 1

        mostPopular = [(gameCount[x], x) for x in gameCount]
        mostPopular.sort()
        mostPopular.reverse()

        return1 = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            return1.add(i)
            if count > totalPlayed * threshold: break
        return return1


# %%
model = PlayPredictor()
model.fit(train, K=6, iters=200)

error = 0
balanced_valid = get_balanced_set(dataset, valid)
for user, game, review in balanced_valid:
    pred = model.predict(user, game, threshold=0.5)
    if pred != review["played"]:
        error += 1

print(f"PlayPredictor accuracy: ", 1 - error / len(balanced_valid))

# %%
writePredictions("pairs_Played.csv", "predictions_Played.csv", model)

# %% [markdown]
# ### Time Predictor

# %%
from copy import copy

class TimePredictor:
    
    def __init__(self):
        pass

    def fit(self, data, l=5.0, iters=200): # data is an array of (user, game, review) tuples
        reviewsPerUser = defaultdict(list)
        reviewsPerItem = defaultdict(list)

        globalAverage = 0

        for user, game, review in data:
            reviewsPerUser[user].append(review)
            reviewsPerItem[game].append(review)

            globalAverage += review["hours_transformed"]

        globalAverage /= len(data)

        betaU = {}
        betaI = {}
        for u in reviewsPerUser:
            reviews = [r["hours_transformed"] for r in reviewsPerUser[u]]
            betaU[u] = np.mean(reviews)

        for g in reviewsPerItem:
            reviews = [r["hours_transformed"] for r in reviewsPerItem[g]]
            betaI[g] = np.mean(reviews)

        alpha = globalAverage # Could initialize anywhere, this is a guess

        for i in range(iters):

            newAlpha = 0
            for user,game,review in data:
                newAlpha += review["hours_transformed"] - (betaU[user] + betaI[game])
            alpha = newAlpha / len(data)

            for user in reviewsPerUser:
                bu = 0
                for review in reviewsPerUser[user]:
                    item = review["gameID"]
                    bu += review["hours_transformed"] - (alpha + betaI[item])
                betaU[user] = bu / (l + len(reviewsPerUser[user]))
            
            for item in reviewsPerItem:
                bi = 0
                for review in reviewsPerItem[item]:
                    user = review["userID"]
                    bi += review["hours_transformed"] - (alpha + betaU[user])
                betaI[item] = bi / (l + len(reviewsPerItem[item]))
        
        self.alpha = alpha
        self.betaU = betaU
        self.betaI = betaI

    def predict(self, user, game):
        bu = 0
        bi = 0

        if user in self.betaU:
            bu = self.betaU[user]
        
        if game in self.betaI:
            bi = self.betaI[game]

        return self.alpha + bu + bi

# %%
from sklearn.metrics import mean_squared_error

def MSE(y, ypred):
    return mean_squared_error(y, ypred)

model = TimePredictor()
model.fit(train)

y = []
y_pred = []
for user, game, review in valid:
    y_pred.append(model.predict(user, game))
    y.append(review["hours_transformed"])

print(f"TimePredictor MSE: {MSE(y, y_pred)}")

# %%
writePredictions("pairs_Hours.csv", "predictions_Hours.csv", model)


