import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# import shots data
s = pd.read_csv("FULL_Shot_Data.csv")

# Remove penalties & own goals
s.drop(s[s.situation == "Penalty"].index, inplace=True)
s.drop(s[s.result == "OwnGoal"].index, inplace=True)

# Create dummy variables for shot outcome, body part, preceding action and game situation
s = pd.get_dummies(s, columns=['result', 'situation', 'shotType', 'lastAction'])

# Transform location co-ordinate to yards and distance to goalline & centre of the pitch
s['to_centre'] = np.where(s['Y']>=0.5, 74 * (s['Y'] - 0.5), 74 * (0.5 - s['Y']))
s['to_goalline'] = 115 * (1 - s['X'])

# Calculate distance to middle of the goal
s['distance'] = np.sqrt(s['to_centre'] ** 2 + s['to_goalline'] ** 2)

# Calculate relative angle to the centre of the goal
s['angle'] = (90-np.angle(s['to_goalline'] + 1j*s['to_centre'], deg=True))/90

# Add variables for inverse of distance & angle, avoiding null values from dividing by zero
s['inv_distance'] = np.where(s['distance'] > 0, 1/s['distance'], 99999999)
s['inv_angle'] = np.where(s['angle'] > 0, 1/s['angle'], 99999999)

# Create datasets for subsets of shots we will need, these are:
# Footed non-cross open-play shots (reg)
# Headers from crosses (h_cr)
# Footed from crosses (cr)
# Header not cross (h)
# Direct Free Kick (fk)

header = s.shotType_Head == 1
other = s.shotType_OtherBodyPart == 1
cross = s.lastAction_Cross == 1
freekick = s.situation_DirectFreekick == 1

reg=s[~other & ~header & ~cross & ~freekick]
h_cr=s[(other | header) & cross & ~freekick]
cr=s[~other & ~header & cross & ~freekick]
h=s[(other | header) & ~cross & ~freekick]
fk = freekick

# Created footedness pivot & merge into reg shots sub-set
foot = reg.pivot_table(index=['player_id'], values=['shotType_LeftFoot', 'shotType_RightFoot'])
reg = reg.merge(foot, right_index=True, how='left', left_on='player_id')

# Add foot strength rating for each shot
reg['foot_strength']=np.where(reg['shotType_LeftFoot_x'] == 1, reg['shotType_LeftFoot_y'], reg['shotType_RightFoot_y'])

# Set up variables for model on reg
dep_variables = ['distance', 'inv_distance', 'angle', 'inv_angle', 'lastAction_Throughball', 'lastAction_TakeOn', 'lastAction_Save', 'lastAction_Pass', 'lastAction_Aerial', 'lastAction_None', 'foot_strength']
input = reg[dep_variables]
target = reg.result_Goal

# Split data into training & test sets
inpt_train, inpt_test, target_train, target_test = train_test_split(input, target, test_size=0.25, random_state=42)

# Set up testing dataset for Understat comparison
usxg = reg.xG
us_train, us_test, tgt_us_train, tgt_us_test = train_test_split(usxg, target, test_size=0.25, random_state=42)

# Fit model for reg
xg_reg = LogisticRegression(solver='liblinear')
xg_reg.fit(inpt_train, target_train)

# Plot ROC curve For reg
target_pred_proba = xg_reg.predict_proba(inpt_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(target_test,  target_pred_proba)
auc = metrics.roc_auc_score(target_test, target_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)

# Plot ROC curve For Understat on regular shots
fpr, tpr, _ = metrics.roc_curve(tgt_us_test,  us_test)
auc = metrics.roc_auc_score(tgt_us_test, us_test)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()
