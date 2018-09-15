![alt text](https://github.com/ggeop/Recommendation-System/blob/master/imgs/photo_cover-01.png)

## Datasets

### Original Dataset

The dataset we used consists of 'circles' (or “friends” lists') from Facebook. Facebook data was collected from survey participants using this Facebook app. The dataset includes node features (profiles), circles, and ego networks. This dataset consists of 4039 nodes (users) and 88234 edges (friendships between users). 
The dataset was downloaded from https://snap.stanford.edu/data/egonets-Facebook.html. 

#### Transform the graph to undirected

```{py}
#Load the dataset
data=pd.read_csv("facebook_combined.txt",sep=" ", header=None)

#Add column names
data.columns = ["node1", "node2"]

#Transform the graph to undirected
data2=pd.concat([data.node2,data.node1], axis=1)

#Rename the columns in order to merge the columns
data2.columns= ["node1", "node2"]
data=data.append(data2)

#Reset indexes
data = data.reset_index(drop=True)

```

In the original dataset the graph was connected and directed (each edge counts as friendship for both nodes). A directed graph  is a graph in which edges have orientations. So for this project to work, we had to transform the graph into undirected graph in order to perform all the necessary tasks. An undirected graph is a graph in which edges have no orientation. 

In order to achieve this the only thing we had to do was to add the missing edges. For instance, if we have an edge from node1 to node2 we added an edge from node2 to node1 in the file. We did that by using pandas library, and more specific we used the command concat() in order to concatenate the missing edges.
In that way we have created a new dataset in which we have changed the position of the two columns we already have (node1, node2).  Then we renamed the columns again in order to use the append() command to merge the two columns by rows. Since we wanted to use the append() command we had to rename the columns because this command performs the merging based on the indexes of the columns.

### Test Dataset

```
#Create a sample graph dataset
test_data = pd.DataFrame([[5, 2], 
                       [9, 3],
                       [9, 11],
                       [3, 6],
                       [4, 6],
                       [5, 7],
                       [1, 11],
                       [6, 2],
                       [7, 9],
                       [8, 9],
                       [5, 11],
                       [6, 7],
                       [6, 11],
                       [7, 6],
                       [2, 11],
                       [11,2],
                       [2, 5],
                       [2, 7],
                       [7, 2]],
                      columns=["node1", "node2"])

```
Now we have created dictionaries with the id of the node as the key and a set with the friends of the node as the value.

```
friendships={}
#Create friendships dict
for node in [1,2,3,4,5,6,7,8,9,11]:
    #Create a list with the friends of node
    ls=dataset[test_data.node1 == node]['node2'].tolist()

    #Create a dictionary with key the node and value the list
    friendships[node]=ls
    
print(friendships) 

```
The relationships are the following: 1: [11], 2: [11, 5, 7], 3: [6], 4: [6], 5: [2, 7, 11], 6: [2, 7, 11], 7: [9, 6, 2], 8: [9], 9: [3, 11], 11: [2]

After that we created a graphical representation of the above relationships.

![alt text](https://github.com/ggeop/Recommendation-System/blob/master/imgs/SampleDatasetgGraphs.png)

## Local structure based methods for link prediction

### Recommending friends using Common neighbors (friend-of-friend (FoF) method)

Common Neighbor method is one of the simplest techniques used for link prediction. Two nodes are likely to form a link if they have many common neighbors. 
This method of link prediction is also called friend-of-friend link prediction.

So, what we did here was to create a function that takes as input the users, the dataset we want and the target user that we want to make the recommendation. At the beginning, for the target user we found all of its friends.

Then we used the intersection between the friends of the target user and the rest of the users and we created a set. And finally we created the sorted list, with the ability in ties to take the smallest number of user id, with the 10 recommended friends for the target user using the Common Neighbors method.


```
def friendOfFriendScore(users, dataset, target):  
    #Initialize  
    l=list()  
    friendships={}  
  
    #Create friendships dict  
    for node in users:  
        #Create a list with the friends of node  
        ls=dataset[dataset.node1 == node]['node2'].tolist()  
  
        #Create a dictionary with key the node and value the list  
        friendships[node]=ls  
  
    # Initialize a dictionary with the intersections  
    inter={}  
  
    #Intersection between users  
    for j in friendships:  
        if (target != j) and (target not in friendships[j]) :  
                 intersection=(len(set(friendships.get(target)).intersection(set(friendships.get(j)))))  
              
            #Keep intersection into a list
            inter[j]=intersection  
     
    #Create a sorted list, in ties we take the smallest ID  
    lis=sorted(inter.items(), key=lambda value: value[1], reverse=True)
  
    #Final Result  
    return(lis[0:10]);  

```

### Recommending friends using Jaccard coefficient

Jaccard   in 1901 proposed a statistic to compare similarity and diversity of sample sets. 
It is the ratio of common neighbors of nodes x and y to the all neighbors nodes of x and y. 
As a result value of Jaccard index prevents higher degree nodes to have high similarity index with other nodes.
Based on that we created a function that takes as an input again the users, the dataset and the target user. At first, we found the target user’s friends as previous and then we applied the Jaccard coefficient, by dividing the intersection of the target user’s friends with the rest of the users by the equivalent union.

At the end we had again a sorted list, with the ability in ties to take the smallest number of user id, of the 10 recommended friends for the target user using the Jaccard similarity coefficient.

```

def JaccardCoefficientScore(users, dataset, target):
    #Initialize
    l=list()
    friendships={}

    #Create friendships dict
    for node in users:
        #Create a list with the friends of node
        ls=dataset[dataset.node1 == node]['node2'].tolist()

        #Create a dictionary with key the node and value the list
        friendships[node]=ls

    # Initialize a dictionary with the intersections
    inter={}

    #Intersection between users
    for j in friendships:
        if (target != j) and (target not in friendships[j]) :
            
            # Create union
             union=len(set(friendships.get(target)).union(set(friendships.get(j))))
            
            # Check for No zero denominator
            if (union != 0) :
                inter[j]=len(set(friendships.get(target)).intersection(set(friendships.get(j))))/union

    #Create a sorted list, in ties we take the smallest ID
    lis=sorted(inter.items(), key=lambda value: value[1], reverse=True)

    #Final Result
    return(lis[0:10]);

```

### Recommending friends using Adamic/Adar function

Adamic-Adar index proposed by Adamic and Adar is calculated by adding weights to the nodes which are connected to both nodes A and B.
Again we created a function which takes as inputs the users, the dataset and the target user. Then we found all the friends of the target user and put them in a list. Then, we had to apply the Adamic/Adar so we performed the intersection between the target user’s friends and the rest of the users. Then we applied the Adamic/Adar measure by summing the number of neighbors any two users have in common and divide it by the log frequency of their occurrence, in order to weight items that are unique to a few users more than commonly occurring items.

At the end we created again a sorted list, with the ability in ties to take the smallest number of user id, of the 10 recommended friends for the target user according to the Adamic/Adar measure.


```

def AdamicAdarFunctionScore(users, dataset, target):
    #Initialize
    l=list()
    friendships={}

    #Create friendships dict
    for node in users:
        #Create a list with the friends of node
        ls=dataset[dataset.node1 == node]['node2'].tolist()

        #Create a dictionary with key the node and value the list
        friendships[node]=ls

    # Initialize a dictionary with the intersections
    inter={}

    #Intersection between users
    for j in friendships:
        if (target != j) and (target not in friendships[j]) :
            intersection =    set(friendships.get(target)).intersection(set(friendships.get(j)))

            # Adamic and Adar score calculation
            sum = 0
            for k in intersection :
                if (k in friendships.keys()) and (friendships[k] != []) and len(friendships[k]) != 1:
                    sum = sum+1/np.log(len(friendships[k]))

            inter[j]=sum
   
    #Create a sorted list, in ties we take the smallest ID
    lis=sorted(inter.items(), key=lambda value: value[1], reverse=True)

    #Final Result
    return(lis[0:10]);

```

### Recommending Friends with Leicht-Holme-Newman Index (Extra method)

Leicht proposed a measure to define local structure based similarity measure. It is the ratio of common neighbors of nodes a and b to the product of degrees of nodes a and b.
So we created a function with inputs the users, the dataset and the target user. As in previous methods we found the target user’s friends and stored them in a list. Then we performed the intersection between the target user’s friends and the rest of the users and stored them in a set. And finally we calculated the degrees of the two compared users each time and we divided the intersection with that. So now we have our final sorted list, with the ability in ties to take the smallest number of user id, of the 10 recommended friends for the target user according to the Leicht-Holme-Newman method.

```
    
def LeichtHolmeNewmanScore(users, dataset, target):  
        #Initialize  
        l=list()  
        friendships={}  
      
        #Create friendships dict  
        for node in users:  
            #Create a list with the friends of node  
            ls=dataset[dataset.node1 == node]['node2'].tolist()  
      
            #Create a dictionary with key the node and value the list  
            friendships[node]=ls  
      
        # Initialize a dictionary with the intersections  
        inter={}  
      
        #Intersection between users  
        for j in friendships:  
            if (target != j) and (target not in friendships[j]) :  
                intersection=(len(set(friendships.get(target)).intersection(set(friendships.get(j)))))  
                  
                #Calculate the k for j and target  
                k1=len(friendships.get(j))  
                k2=len(friendships.get(target))  
                  
                if (k1 !=0 and k2 !=0):  
                    #Store the intersection in the list inter[]  
                    inter[j]=intersection/(k1*k2)  
         
        #Create a sorted list, in ties we take the smallest ID  
        lis=sorted(inter.items(), key=lambda value: value[1], reverse=True)  
      
        #Final Result  
        return(lis[0:10]);  

```

## Evaluation of the recommendation system

### Compute the average similarity

```

# Create users list
users=list(range(0,4038))

#Initialization
s1=[]
s2=[]
s3=[]

for i in list(range(100,4100,100)):
    
    #Run the functions
    fofList=friendOfFriendScore(users, data,i)
    JaccardList=JaccardCoefficientScore (users, data,i)
    AdamicAdarList=AdamicAdarFunctionScore (users, data,i)
    
    #Similarity Percentage of FoF and Jaccard
    s1.append(len(set(fofList).intersection(set(JaccardList)))*10)
    
    #Similarity Percentage of FoF and Adamic and Adar
    s2.append(len(set(fofList).intersection(set(AdamicAdarList)))*10)
    
    #Similarity Percentage of Jaccard and Adamic and Adar
    s3.append(len(set(AdamicAdarList).intersection(set(JaccardList)))*10)

#Average Similarity (%)
print("The average similarity of FoF & Jaccard is:",np.mean(s1),"%")
print("The average similarity of FoF & Adamic Adar is:",np.mean(s2),"%")
print("The average similarity of Adamic Adar & Jaccard is:",np.mean(s3),"%")

```

After a few minutes of computations finally the results are:

o	The average similarity of FoF & Jaccard is: 55.5 %
o	The average similarity of FoF & Adamic/Adar is: 90.75 %
o	The average similarity of Adamic/Adar & Jaccard is: 57.0 %

### Forecast Recommendations

#### Evaluation Function
In this stage we have to estimate the quality of the recommendation methods. We create a function (evaluationFunction()) which computes the strength of the connection between two nodes. In more details, we insert two already friends of our network and the function removes this connection from the dataset. After the connection is dropped, the algorithm searches for every method if one of the two nodes (ex. F1) exists in the list of the second node (ex. F2). We do the same process in both F1 and F2. Also, we would like to mention that if a node does not exist in the recommendation list of the other node we exclude this relationship.

#### Score Calculation
The score for each algorithm is calculated according to the position of the list. Also, we take the average value of the position for both F1 and F2. The higher the score is, the higher the quality of the algorithm.

```

def evaluationFunction(dataset,users,F1,F2):

    ####Remove the relationship
    
    #First we find the connection F1-F2
    l1=dataset[dataset.node2 == F1].index
    l2=dataset[dataset.node1 == F2 ].index
    rm1=set(l1).intersection(set(l2))
    
    #Then we find the connection F2-F1
    l1=dataset[dataset.node2 == F2 ].index
    l2=dataset[dataset.node1 == F1].index
    rm2=set(l1).intersection(set(l2))

    #We create the union
    rm=rm1.union(rm2)
   
    
    #Remove the elements of the set rm
    for i in rm:
        dataset=dataset.drop(i)
    
    ###FoF (friend-of-friend)
    if ((F1 in friendOfFriend(users, dataset,F2)) and (F2 in friendOfFriend(users, dataset,F1))): 
        
        #Compute the recommendations for F1
        Friend1=10 - friendOfFriend(users, dataset,F1).index(F2)

        #Compute the resommentdations for F2
        Friend2=10 - friendOfFriend(users, dataset,F2).index(F1)


        ####Compute the score
        scoreFoF=(Friend1+Friend2)/2
        
    else:
        return(None);
    
    ###Jaccard
    if ((F1 in JaccardCoefficient(users, dataset,F2)) and (F2 in JaccardCoefficient(users, dataset,F1))): 
        
        #Compute the recommendations for F1
        Friend1=10 - JaccardCoefficient(users, dataset,F1).index(F2)

        #Compute the resommentdations for F2
        Friend2=10 - JaccardCoefficient(users, dataset,F2).index(F1)

        ####Check if either of these does not exist

        ####Compute the score
        scoreJaccard=(Friend1+Friend2)/2
        
    else:
        return(None);
    
    ###AdamicAdar
    if ((F1 in AdamicAdarFunction(users, dataset,F2)) and (F2 in AdamicAdarFunction(users, dataset,F1))):
        
        #Compute the recommendations for F1
        Friend1=10 - AdamicAdarFunction(users, dataset,F1).index(F2)

        #Compute the resommentdations for F2
        Friend2=10 - AdamicAdarFunction(users, dataset,F2).index(F1)

        ####Check if either of these does not exist

        ####Compute the score
        scoreAdamicAdar=(Friend1+Friend2)/2
        
    else:
        return(None); 
          
 
    return(scoreFoF,scoreJaccard,scoreAdamicAdar);

```

#### Iteration Function
In order to have more accurate results we should run the algorithm more than once. So, we created the algorithm (finalscore()) in order to recall the evaluation function many times. Specifically, this function takes a random index from the original dataset. Then we call the evaluation function in order to create a score for each relationship; the outputs of the evaluation function are stored in a list. Finally, after all that repetitions, we calculate the average score for each method.

```

#Function for iterations
def finalScore(dataset,users, n):
    eval_scores=[]

    #We want to have 100 succesful iterations in all methods
    while i<n :
        
        #Random F1-F2 Relationship
        index=sample(range(len(dataset)),1)[0]
        friend1=dataset.iloc[index].node1
        friend2=dataset.iloc[index].node2
        
        evaluationOutput=evaluationFunction(dataset,users,friend1,friend2)
        
        if (evaluationOutput!= None):
        i +=1
            eval_scores.append(evaluationFunction(dataset,users,friend1,friend2))
        
    #Final Score
    if eval_scores != [] :
        scores=eval_scores      
        
    return(scores);

```

### Summary and Discussion
All of the methods that we have used for the prediction belong to the greater family of Local structure based methods of link prediction. The methods that belong to this family computes the similarity score based on common neighbors which gives an accurate measure to know link structure arising between nodes. Such measure is computed only between nodes of path length 2 and not beyond that. As a result some interesting and potential links may be missed and also as a matter of fact it will be difficult and time consuming for computing similarity score for all nodes in network. 

Having that in mind, since all of our methods belongs to the same family we did not expect any significant deviance among them. After we compared the methods the assumption we had in mind about all the methods being close, came to life. So, as you can see from the graph below the three methods were really close, with the Adamic/Adar method to have the best score and the other two following. We also had in mind that the results may differ each time we run the algorithm, something that we did not want, so we had to test it. After running the algorithm several times we ended up to the conclusion that the Adamic/Adar method comes always as the best method among the three of them but the other two may change positions each time. So, as a result the Adamic/Adar method is the best for the recommendation of friends, with a slightly better score than the other two, and the Jaccard and FoF methods are coming in second and third place with a negligible deviance among them. About the extra method we used, Leicht-Holme&Newman, it was always the worst among the other three for the prediction.

![alt text](https://github.com/ggeop/Recommendation-System/blob/master/imgs/MethodsComparison.png)

### 2nd Evaluation System

We created an alternative evaluation system in order to increase the accuracy. Specifically, we took a target user and we removed one by one all of its friends. Hence, we calculated the number of friends that were suggested and then we divided it with the total initial number of friends. This computations are hard and intensive, so we did this procedure for 2-3 different targets. Also we created the following figure (see the Figure 8 - 2nd Evaluation System Interpretation) in order to explain the procedure of this evaluation method.

![alt text](https://github.com/ggeop/Recommendation-System/blob/master/imgs/Evaluation%20System%20v2.png)
```
def finalScore2(dataset,users, n):
    
    eval_scores=[]
    
    #Create friendships dict
    friendships={}
    for node in users:
        #Create a list with the friends of node
        ls=dataset[dataset.node1 == node]['node2'].tolist()

        #Create a dictionary with key the node and value the list
        friendships[node]=ls

    
for l in list(range(0,n)):
        
        target=sample(users,1)[0] #take 1 random user
        
        #Initialize the scores
        scorefof = 0
        scoreJaccard = 0
        scoreAdamicAdar = 0
        friends=[]
        
        for i in friendships.get(target):
            friends.append([target,i])
            
        for i in friends:
            evaluationOutput=evaluationFunction(dataset,users,i[0],i[1])

            if (evaluationOutput!= None):
                if evaluationOutput[0]!= 0:
                    scorefof +=1
                if evaluationOutput[1]!= 0:
                    scoreJaccard +=1
                if evaluationOutput[2]!= 0:    
                    scoreAdamicAdar +=1
        
        if len(friends) !=0 :
            eval_scores.append([scorefof/len(friends),scoreJaccard/len(friends),scoreAdamicAdar/len(friends)])
            
        #Final Print
        if eval_scores != [] :
            #Calculate the average value
            scores=np.mean(eval_scores,axis=0)

    print("FoF score is: ", round(scores[0],4))
    print("Jaccard score is: ", round(scores[1],4))
    print("Adamic Adar score is: ", round(scores[2],4))
        
        
    return;  

```

Finally, the output ranking score is approximately the same with the first evaluation system. This method takes into consideration all the local friends of a target user so it need a lot of time if we want to calculate the score in a many users. Also, if we take into consideration only few users maybe the results could be biased but our system computation power is very limited.
