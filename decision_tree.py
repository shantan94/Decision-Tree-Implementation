import pandas,random,math,copy,sys,pprint,pydot

graph=pydot.Dot(graph_type='graph')

def entropy(data):
	total=len(data)
	dict1={}
	for row in data:
		if row[len(row)-1] not in dict1:
			dict1[row[len(row)-1]]=1
		else:
			dict1[row[len(row)-1]]+=1
	entropy=0.0
	for key,val in dict1.items():
		entropy+=(val/total)*math.log2(val/total)
	return -entropy

def data_split(data,index,value):
	split_data=[]
	for row in data:
		if row[index]==value:
			temp_data=row[:index]
			for i in range(index+1,len(row)):
				temp_data.append(row[i])
			split_data.append(temp_data)
	return split_data

def information_gain(data):
	init_entropy=entropy(data)
	info_gain=float("-inf")
	index=0
	for i in range(len(data[0])-1):
		values=[]
		for row in data:
			values.append(row[i])
		unique=set()
		for row in values:
			unique.add(row)
		new_entropy=0.0
		for row in unique:
			new_values=data_split(data,i,row)
			new_entropy+=(len(new_values)/len(data))*entropy(new_values)
		info_gain_new=init_entropy-new_entropy
		if info_gain_new>info_gain:
			info_gain=info_gain_new
			index=i
	return index

def peak(classes):
	dict1={}
	classid=""
	for row in classes:
		if row not in dict1:
			dict1[row]=1
		else:
			dict1[row]+=1
	peak=-sys.maxsize
	for key,val in dict1.items():
		if peak<val:
			peak=val
	for key,val in dict1.items():
		if peak==val:
			classid=key
	return classid

def make_tree(data,labels):
	classes=[]
	for row in data:
		classes.append(row[len(row)-1])
	count=0
	for value in classes:
		if value==classes[0]:
			count+=1
	if count==len(classes):
		return classes[0]
	if len(data[0])==1:
		return peak(classes)
	bestchoice=information_gain(data)
	bestlabel=labels[bestchoice]
	decisiontree={}
	decisiontree.update({bestlabel:{}})
	labels.pop(bestchoice)
	values=[]
	for row in data:
		values.append(row[bestchoice])
	unique=set()
	for i in range(len(values)):
		unique.add(values[i])
	for row in unique:
		labelcopy=copy.deepcopy(labels)
		insertvalue=data_split(data,bestchoice,row)
		treevalue=make_tree(insertvalue,labelcopy)
		decisiontree[bestlabel][row]=treevalue
	return decisiontree

def train_test_split(data,random_state,train_split):
	# random.Random(random_state).shuffle(data)
	random.shuffle(data)
	split_range=int(len(data)*train_split)
	train_data=data[:split_range]
	test_data=data[split_range:]
	return train_data,test_data

def predict(tree,labels,data,indexes):
	rootnode=list(tree)[0]
	dict1=tree[rootnode]
	index=labels.index(rootnode)
	key=data[index]
	value=indexes[0]
	try:
		value=dict1[key]
	except:
		if labels[0]=="sepal length in cm" or labels[0]=="sepal length in cm" or \
		labels[0]=="petal length in cm" or labels[0]=="petal width in cm":
			for key1 in dict1:
				if (int)(key)==(int)(key1):
					value=dict1[key1]
				elif (int)(key+1)==(int)(key1):
					value=dict1[key1]
				elif (int)(key-1)==(int)(key1):
					value=dict1[key1]
		else:
			pass
	if isinstance(value,dict):
		prediction=predict(value,labels,data,indexes)
	else:
		prediction=value
	return prediction

def predict_with_confusion_matrix(tree,labels,indexes,data):
	count=0
	ind_indexes=[]
	for index in indexes:
		ind_indexes.append(index)
	temp_matrix=[[0 for i in range(len(indexes))] for j in range(len(indexes))]
	confusion_matrix=pandas.DataFrame(temp_matrix,index=indexes,columns=indexes)
	total=len(data)
	for row in data:
		temp_arr=row[:len(row)-1]
		value=predict(tree,labels,temp_arr,ind_indexes)
		if value==row[len(row)-1]:
			count+=1
		confusion_matrix[row[len(row)-1]][value]+=1
	accuracy=count/total
	print("The Confusion Matrix is: \n"+str(confusion_matrix)+"\n")
	return accuracy*100

def cv_split(train_data,folds):
	main_set=list()
	temp_set=list(train_data)
	size=int(len(train_data)/folds)
	for i in range(folds):
		fold=list()
		while len(fold)<size:
			fold.append(temp_set.pop())
		main_set.append(fold)
	return main_set

def cross_validate(data,labels,indexes,folds):
	ind_indexes=[]
	for index in indexes:
		ind_indexes.append(index)
	split_set=list(cv_split(data,folds))
	accuracy_scores=[]
	for i in range(len(split_set)):
		train_temp=[]
		test_temp=[]
		test_temp+=split_set[i]
		for j in range(len(split_set)):
			if i!=j:
				train_temp+=split_set[j]
		predlabels=copy.deepcopy(labels)
		decisiontree=make_tree(train_temp,predlabels)
		count=0
		total=len(test_temp)
		for row1 in test_temp:
			value=predict(decisiontree,labels,row1,ind_indexes)
			if value==row1[len(row1)-1]:
				count+=1
		accuracy=(float)(count)/total
		accuracy_scores.append(accuracy)
	return accuracy_scores

def make_graph(tree1,parent_name,child_name):
	tree1.append(parent_name)
	tree1.append(child_name)
	if len(tree1)==2:
		edge=pydot.Edge(str("root: "+parent_name),str(child_name))
		graph.add_edge(edge)
		return
	# if ((parent_name in tree1) and (child_name in tree2)) and ((child_name in tree1) and (parent_name in tree2)):
	edge=pydot.Edge(str(parent_name),str(child_name))
	graph.add_edge(edge)

def visualizetree(tree1,node,parent=None):
	for key,val in node.items():
		if isinstance(val,dict):
			if parent:
				make_graph(tree1,parent,key)
			visualizetree(tree1,val,key)
		else:
			make_graph(tree1,parent,key)
			make_graph(tree1,key,val)

def main():
	line=""
	dataset_file=sys.argv[1]
	dataset_names=sys.argv[2]
	with open(dataset_names,'r') as f:
		line=f.read()
	splitline=line.split(",")
	names=[]
	for data in splitline:
		names.append(data)
	df=pandas.read_csv(dataset_file,names=names,na_values=['?'])
	# df=df.apply(lambda column:column.fillna(column.value_counts().index[0]))
	df=df.dropna()
	data=[]
	indexes=[]
	for index,row in df.iterrows():
		tempdata=[]
		for i in range(len(names)):
			tempdata.append(row[names[i]])
			if i==len(names)-1:
				indexes.append(row[names[i]])
		data.append(tempdata)
	labels=splitline
	predlabels=copy.deepcopy(labels)
	train_data,test_data=train_test_split(data,random_state=100,train_split=0.8)
	decisiontree=make_tree(train_data,labels)
	pprint.PrettyPrinter(compact=True).pprint(decisiontree)
	with open("decisiontree.txt","wt") as out:
		pprint.pprint(decisiontree,stream=out)
	tree1=[]
	visualizetree(tree1,decisiontree)
	graph.write_png('decisiontree.png')
	indexes=set(indexes)
	dt_accuracy=predict_with_confusion_matrix(decisiontree,predlabels,indexes,test_data)
	print("Decision Tree Classifier accuracy is: "+str(round(dt_accuracy,2))+"%")
	scores=cross_validate(data,labels,indexes,folds=4)
	print("Cross Validation splits are:")
	print(scores)
	result=0.0
	for value in scores:
		result+=value
	result/=len(scores)
	print("Cross Validation accuracy is: "+str(round(result*100,2))+"%")

if __name__=='__main__':
	main()