import matplotlib
import os
import seaborn as sns; #sns.set()
import matplotlib.pyplot as plt
import pickle
import ipdb
import pandas as pd


def plot_line(folder, label):

	x = list()
	y = list()

	for i in range(1,11):
		fname = os.path.join(folder, str(i), "log.p")
		#print(fname)
		log = pickle.load(open(fname, "rb"))
		for ent in log:
			x.append(ent[1])
			y.append(ent[0])

	df = pd.DataFrame(dict(steps=x,
						   reward=y))


	ax = sns.lineplot(x="steps", y="reward", label=label,legend="brief", data=df)

sns.set_style("whitegrid")


ename = "pusher"
#ename = "reacher"
#ename = "cartpole"

#plot_line("logs/"+ename+"/epochs05", ename)


#plot_line("logs/"+ename+"/epochs05", "5 epochs")
#plot_line("logs/"+ename+"/epochs10", "10 epochs")
#plot_line("logs/"+ename+"/epochs15", "15 epochs")
#plot_line("logs/"+ename+"/epochs20", "20 epochs")
#plot_line("logs/"+ename+"/epochs25", "25 epochs")
#plot_line("logs/"+ename+"/epochs30", "30 epochs")
#plot_line("logs/"+ename+"/epochs35", "35 epochs")
#plot_line("logs/"+ename+"/epochs40", "40 epochs")
#plot_line("logs/"+ename+"/epochs100", "100 epochs")
#plot_line("logs/"+ename+"/epochs200", "200 epochs")

#plot_line("logs/"+ename+"/epochs05", "5 numnets")
#plot_line("logs/"+ename+"/epochs10", "10 epochs")
#plot_line("logs/"+ename+"/epochs15", "15 epochs")
#plot_line("logs/"+ename+"/epochs20", "20 epochs")
#plot_line("logs/"+ename+"/epochs25", "25 epochs")
#plot_line("logs/"+ename+"/epochs30", "30 epochs")
#plot_line("logs/"+ename+"/epochs35", "35 epochs")
#plot_line("logs/"+ename+"/epochs40", "40 epochs")
#plot_line("logs/"+ename+"/numnets10", "10 numnets")

#plot_line("logs/"+ename+"/popsize200", "200 popsize")
#plot_line("logs/"+ename+"/epochs05", "400 popsize")
#plot_line("logs/"+ename+"/popsize800", "800 popsize")
#plot_line("logs/"+ename+"/popsize1000", "1000 popsize")


#plot_line("logs/"+ename+"/epochs05", "cem 5")
#plot_line("logs/"+ename+"/max_iters10", "cem 10")
#plot_line("logs/"+ename+"/max_iters15", "cem 15")


#plot_line("logs/"+ename+"/epochs05", "epochs 5 cem 5")
#plot_line("logs/"+ename+"/max_iters15", "epochs 5 cem 15")
#plot_line("logs/"+ename+"/epochs100", "epochs 100 cem 5")
#plot_line("logs/"+ename+"/epochs100cem15", "epochs 100 cem 15")

#plot_line("logs/"+ename+"/popsize1000elite100", "1000 pop 100 elite")



#plot_line("logs/"+ename+"/epochs05", "epochs 5 cem 5 pop 400")
#plot_line("logs/"+ename+"/max_iters15", "epochs 5 cem 15 pop 400")
#plot_line("logs/"+ename+"/epochs100", "epochs 100 cem 5 pop 400")
#plot_line("logs/"+ename+"/epochs100cem15", "epochs 100 cem 15 pop 400")
#plot_line("logs/"+ename+"/popsize1000epoch100cem15", "epochs 100 cem 15 pop 1k")
#plot_line("logs/"+ename+"/popsize1000", "epochs 5 cem 5 pop 1k")


plot_line("logs/"+ename+"/epochs05", "PETS")
plot_line("logs/"+ename+"/epochs100cem15", "PETS with epochs 100 cem 15")


#sns.axes_style("whitegrid")


plt.title(ename)

plt.legend(loc='lower right', fontsize=12)
matplotlib.pyplot.xticks(fontsize=12)
matplotlib.pyplot.yticks(fontsize=12)

#matplotlib.pyplot.xlim(0, 10000)
#matplotlib.pyplot.xlim(0, 2000)

plt.show()
