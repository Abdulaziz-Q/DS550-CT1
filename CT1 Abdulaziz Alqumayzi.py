# importing needed packages

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# read the dataset
df = pd.read_csv('crx.csv')

# renaming the targeted column
df = df.rename(columns={"A1": "Target"})

# selecting the numeric features
features = ['A2','A3', 'A8','A11','A14','A15']
# separating out the features
x = df.loc[:, features].values
# separating out the target
y = df.loc[:,['Target']].values
# standardizing the features
x = StandardScaler().fit_transform(x)

# build the PCA model
pca = PCA(n_components=2)
pc = pca.fit_transform(x)
p_df = pd.DataFrame(data = pc, columns = ['First principal component', 'Second principal component'])

# concatenating the model with the target
final_df = pd.concat([p_df, df[['Target']]], axis = 1)

# visualizing the model
fig = plt.figure(figsize = (18,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('First principal component', fontsize = 12)
ax.set_ylabel('Second principal component', fontsize = 12)
ax.set_title('PCA of 2 components', fontsize = 25)
targets = ['a','b']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = final_df['Target'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'First principal component']
               , final_df.loc[indicesToKeep, 'Second principal component']
               , c = color
               , s = 50
               , alpha=0.5)

ax.legend(targets)
ax.grid()

# print the explained variance
print(pca.explained_variance_ratio_)

'We can convert 6-dimensional space into 2-dimensional space, ' \
'We lose some of the variance (information) when we do this. ' \
'By using the attribute explained_variance_ratio_, ' \
'we can see that the first principal component contains 32.43% of the variance ' \
'and the second principal component contains 17.88% of the variance. ' \
'Together, the two components contain 50.31% of the information. which is losing much information in this model.'

