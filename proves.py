import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# relMatrix
data = {'movie_title': ['Film1', 'Film2', 'Film3'],
        'User1': [3, 4, 5],
        'User2': [5, 4, 3],
        }
relMatrix = pd.DataFrame(data, columns=['movie_title', 'User1', 'User2'])
relMatrix = relMatrix.set_index("movie_title")
print (relMatrix)

# ratings_pd
data = {'movie_title': ['Film1', 'Film2', 'Film3'],
        'user_id': ['User1', 'User2', 'User3'],
        'user_rating': [3, 4, 5],
        }
ratings_pd = pd.DataFrame(data, columns=['movie_title', 'user_id', 'user_rating'])
print (ratings_pd)
print(ratings_pd['user_id'] == 'User1')
print(ratings_pd['movie_title'] == 'Film1')
print(((ratings_pd['movie_title'] == 'Film1') & (ratings_pd['movie_title'] == 'Film1')).any())
print(ratings_pd[ratings_pd['user_id'] == 'User1', 'movie_title'])

users = [2, 3, 4]
scores = {'x': users}
listModelings = list()
listModelings.append("average")
listModelings.append("fai")
for modelingStrategy in listModelings:
    scores[modelingStrategy] = list()
print(scores)


# import matplotlib.pyplot as plt
scores = {"x": [2, 3, 4, 5, 6, 7, 8], "similar": [0.09108433382897535, 0.08366990209112672, 0.08219682886540265, 0.08104236878305564, 0.08103274912388536, 0.07770855816755527, 0.07612180849743805], "distinct": [0.09176233711991354, 0.08295097189303509, 0.07939715124278399, 0.07722128191351614, 0.07747930890666331, 0.0743117410960038, 0.07672339624641669],
          "random": [0.09176233711991354, 0.08295097189303509, 0.079397151242784, 0.07722128191351614, 0.07747930890666332, 0.0743117410960038, 0.07672339624641669]}
similarities = ["random", "distinct", "similar"]
for similarity in similarities:
    plt.plot(scores['x'], scores[similarity], marker='o', markersize=3, linewidth=1, label=similarity)
# plt.fill_between(scores['x'], np.array(scores[modelingStrategy]) - np.array(pearsons), np.array(scores[modelingStrategy]) + np.array(pearsons), alpha=0.5, facecolor=palette(color))

plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09108433382897535, 0.08366990209112672, 0.08219682886540265, 0.08104236878305564,
                                 0.08103274912388536, 0.07770855816755527, 0.07612180849743805], marker='o', markersize=3, linewidth=1, label="similar")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.08985304934655446, 0.08162806070902412, 0.08023637170842311, 0.0786276337903283,
                                 0.07938991838554824, 0.07845224406191638, 0.07447430777347556], marker='o', markersize=3, linewidth=1, label="random")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09176233711991354, 0.08295097189303509, 0.079397151242784, 0.07722128191351614,
                                 0.07747930890666332, 0.0743117410960038, 0.07672339624641669], marker='o', markersize=3, linewidth=1, label="distinct")
plt.title("ARM's NDCG scores Vs similarities across users")
plt.xlabel('users per group')
plt.ylabel("ARM's NDCG scores")
plt.legend()
plt.show()

plt.plot([2, 3, 4, 5, 6, 7, 8], [0.39006342494714585, 0.3987341772151898, 0.39522821576763484, 0.39522821576763484,
                                 0.37370600414078675, 0.38951695786228163, 0.38306451612903225], marker='o', markersize=3, linewidth=1, label="similar")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.399364406779661, 0.409853249475891, 0.4023109243697479, 0.4216494845360826,
                                 0.39832285115303984, 0.404933196300103, 0.3801229508196721], marker='o', markersize=3, linewidth=1, label="random")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3845338983050847, 0.38924050632911417, 0.3940677966101695, 0.3802083333333335,
                                 0.36897274633123683, 0.39232409381663097, 0.3686440677966102], marker='o', markersize=3, linewidth=1, label="distinct")
plt.title("ARM's Z-Recall scores Vs similarities across users")
plt.xlabel('users per group')
plt.ylabel("ARM's Z-Recall scores")
plt.legend()
plt.show()

paths = ["groups_movielens100k_distinct_2.txt", "groups_movielens100k_random_2.txt", "groups_movielens100k_similar_2.txt"]

with open("groups_movielens100k_distinct_2.txt", "r") as read_file:
    groups = json.load(read_file)
distinct_pearsons = list()
for i in range(len(groups)):
    distinct_pearsons.append(groups[i]["pearsons"][0])
technique_ = ["distinct"] * len(distinct_pearsons)
pearsons = pd.DataFrame(data=list(zip(distinct_pearsons, technique_)), columns=["pearsons", "technique"])

with open("groups_movielens100k_random_2.txt", "r") as read_file:
    groups = json.load(read_file)
random_pearsons = list()
for i in range(len(groups)):
    random_pearsons.append(groups[i]["pearsons"][0])
technique_ = ["random"] * len(random_pearsons)
pearsons2 = pd.DataFrame(data=list(zip(random_pearsons, technique_)), columns=["pearsons", "technique"])
pearsons = pd.concat([pearsons, pearsons2])

with open("groups_movielens100k_similar_2.txt", "r") as read_file:
    groups = json.load(read_file)
similar_pearsons = list()
for i in range(len(groups)):
    similar_pearsons.append(groups[i]["pearsons"][0])
technique_ = ["similar"] * len(similar_pearsons)
pearsons2 = pd.DataFrame(data=list(zip(similar_pearsons, technique_)), columns=["pearsons", "technique"])
pearsons = pd.concat([pearsons, pearsons2])

sns.boxplot(x=pearsons["technique"], y=pearsons["pearsons"])
plt.show()


plt.plot([2, 3, 4, 5, 6, 7, 8], [2.03, 1.9, 1.83, 1.86, 1.73, 1.70, 1.69], marker='o', markersize=3, linewidth=1, label="merging individual recommendations")
plt.plot([2, 3, 4, 5, 6, 7, 8], [2.13, 1.94, 1.84, 1.86, 1.84, 1.73, 1.76], marker='o', markersize=3, linewidth=1, label="aggregating individual preferences")
plt.title("Elapsed time Vs architecture considered")
plt.xlabel('users per group')
plt.ylabel("Elapsed time (min)")
plt.legend()
plt.show()

plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09108433382897535, 0.08366990209112672, 0.08219682886540265, 0.08104236878305564, 0.08103274912388536,
                                 0.07770855816755527, 0.07612180849743805], marker='o', markersize=3, linewidth=1, label="merging individual recommendations")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09229127652319725, 0.08505153160542761, 0.08206035539727437, 0.08238872117346778, 0.0815844952203215,
                                 0.07813570215025542, 0.07553696372604633], marker='o', markersize=3, linewidth=1, label="aggregating individual preferences")
plt.title("Accuracy Vs architecture considered")
plt.xlabel('users per group')
plt.ylabel("NDCG")
plt.legend()
plt.show()

plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09649854388250764, 0.08908582650718193, 0.08325261704981785, 0.07970048817917522, 0.07681841607232119,
                                 0.07529563045415276, 0.0701994879494208], marker='o', markersize=3, linewidth=1, label="p = 0.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09466441427075876, 0.08793346376213843, 0.08273066635310684, 0.08003116604671938, 0.08037390436170416,
                                 0.0775910734434243, 0.07571272110173705], marker='o', markersize=3, linewidth=1, label="p = 1")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09322027669614076, 0.085856045977389, 0.08273583101225342, 0.08075358670305571, 0.08067241270156685,
                                 0.07826848460598815, 0.07592415988332687], marker='o', markersize=3, linewidth=1, label="p = 1.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.09229127652319725, 0.08505153160542761, 0.08206035539727437, 0.08238872117346778, 0.0815844952203215,
                                 0.07813570215025542, 0.07553696372604633], marker='o', markersize=3, linewidth=1, label="p = 2")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.08782189961153482, 0.08221724009488789, 0.08074258554556143, 0.08014918895220209, 0.07913164315848993,
                                 0.07687567345557197, 0.07335072665488375], marker='o', markersize=3, linewidth=1, label="p = 4")
plt.title("Accuracy Vs parameter P")
plt.xlabel('users per group')
plt.ylabel("NDCG")
plt.legend()
plt.show()


plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3898305084745763, 0.4124472573839661, 0.4184322033898305, 0.4458333333333337, 0.42138364779874204,
                                 0.446695095948827, 0.4184322033898305], marker='o', markersize=3, linewidth=1, label="p = 0.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3919491525423729, 0.4029535864978901, 0.4014830508474576, 0.407291666666667, 0.389937106918239,
                                 0.39765458422174826, 0.3781779661016949], marker='o', markersize=3, linewidth=1, label="p = 1")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3845338983050847, 0.4008438818565402, 0.3877118644067797, 0.3968750000000001, 0.3763102725366876,
                                 0.3933901918976546, 0.3644067796610169], marker='o', markersize=3, linewidth=1, label="p = 1.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3855932203389831, 0.388185654008439, 0.3845338983050847, 0.3760416666666668, 0.37106918238993714,
                                 0.3944562899786779, 0.375], marker='o', markersize=3, linewidth=1, label="p = 2")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.3877118644067797, 0.3924050632911394, 0.3930084745762712, 0.39583333333333326, 0.38364779874213834,
                                 0.3923240938166311, 0.3940677966101695], marker='o', markersize=3, linewidth=1, label="p = 4")
plt.title("Fairness Vs parameter P")
plt.xlabel('users per group')
plt.ylabel("Z-Recall")
plt.legend()
plt.show()

plt.plot([2, 3, 4, 5, 6, 7, 8], [0.07530154019800044, 0.05806096850093587, 0.04936380135130317, 0.04350727498985336, 0.040527674685180926,
                                 0.036926596877439696, 0.034582635281720356], marker='o', markersize=3, linewidth=1, label="p = 0.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.0742406414382431, 0.05777184357174177, 0.04996224475507511, 0.04429369687443369, 0.043861923365014745,
                                 0.04068998805656772, 0.038900753531508535], marker='o', markersize=3, linewidth=1, label="p = 1")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.07340699641502649, 0.05681073194220203, 0.050665257769856184, 0.045662114458345386, 0.04481697113926543,
                                 0.04087713429761572, 0.03949599770957943], marker='o', markersize=3, linewidth=1, label="p = 1.5")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.0730389870524696, 0.05660701236277104, 0.05080119295084847, 0.0475499005718407, 0.04580015549319108,
                                 0.040810734489150555, 0.039039765486597273], marker='o', markersize=3, linewidth=1, label="p = 2")
plt.plot([2, 3, 4, 5, 6, 7, 8], [0.0699072008602198, 0.05505036288003539, 0.050681346686094975, 0.0466092727558221, 0.044609866136306416,
                                 0.04062595886778805, 0.03793956506975364], marker='o', markersize=3, linewidth=1, label="p = 4")
plt.title("Overall Fairness and accuracy Vs parameter P")
plt.xlabel('users per group')
plt.ylabel("BNDCG")
plt.legend()
plt.show()


plt.plot([0.5, 1, 1.5, 2, 4], [0.08155014429922534, 0.08271962990565555, 0.0824901139399601, 0.08243557797085575, 0.08004127963901884], marker='o', markersize=3, linewidth=1, label="NDCG")
plt.xlabel('parameter P')
plt.ylabel("NDCG")
plt.legend()
plt.show()


plt.plot([0.5, 1, 1.5, 2, 4], [0.42186489281701517, 0.3956353019708672, 0.3862959840948234, 0.38212570166968407, 0.39128548925378037], marker='o', markersize=3, linewidth=1, label="Z-Recall")
plt.xlabel('parameter P')
plt.ylabel("Z-Recall")
plt.legend()
plt.show()


plt.plot([0.5, 1, 1.5, 2, 4], [0.04832435598349054, 0.049960155941797806, 0.05024788624741296, 0.05052110691526697, 0.049346224750860056], marker='o', markersize=3, linewidth=1, label="Z-Recall")
plt.xlabel('parameter P')
plt.ylabel("bndcg")
plt.legend()
plt.show()

amazonReviews = amazonReviews.map(lambda x: {
    "movie_title": x["data/product_id"],
    "user_id": x["data/user_id"],
    "user_rating": x["data/star_rating"]
})
