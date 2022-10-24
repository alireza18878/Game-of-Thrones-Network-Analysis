#!/usr/bin/env python
# coding: utf-8

# # **Game of Thrones Network Analysis**

# In[1]:


import warnings
warnings.filterwarnings('ignore')     # to avoid warning messages


# ## **Importing the libraries**

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import networkx as nx

#from decorator import decorator #commented to see if it is required

from networkx.utils import create_random_state, create_py_random_state

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

# Remove scientific notations and display numbers with 2 decimal points instead
pd.options.display.float_format = '{:,.2f}'.format        

# Update the default background style of the plots
sns.set_style(style='darkgrid')

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly

import plotly.express as px
init_notebook_mode(connected=True)


# ## **Loading the raw data**

# In[3]:


os.listdir("raw_data_books/")


# - **There are 5 raw files** (one for each book)
# 
# 

# In[4]:


book1 = pd.read_csv("raw_data_books/book1.csv")


# ## **Checking the first few observations**

# In[5]:


book1.head()


# This is an example of an **Undirected Graph.**

# **Let's load all the files and combine them together**

# In[6]:


book2 = pd.read_csv("raw_data_books/book2.csv")

book3 = pd.read_csv("raw_data_books/book3.csv")

book4 = pd.read_csv("raw_data_books/book4.csv")

book5 = pd.read_csv("raw_data_books/book5.csv")


# In[7]:


books = [book1, book2, book3, book4, book5]

books_combined = pd.DataFrame()

for book in books:
    books_combined = pd.concat([books_combined, book])

# Grouping the data by Person 2 and Person 1 to avoid multiple entries with the same characters 
books_combined = books_combined.groupby(["Person 2", "Person 1"], as_index = False)["weight"].sum()


# ## **Descriptive Analytics** 

# In[8]:


books_combined.info()


# In[9]:


books_combined.describe()


# **Observations:**
# 
# - There are **2823 edges** in total, or 2823 co-occurrences of characters.
# - The **minimum weight is 3** (meaning every co-occurrence pair has been observed at least thrice), and the **maximum weight is 334**.
# - The **mean weight is 11.56**, meaning that on average, two co-occurring characters are mentioned around 12 times together. **The median of 5** also implies that **it is the maximum weight which is more likely the outlier,** which is also affirmed by the fact that 75% of the weight values are 11 or lower.

# In[10]:


books_combined[books_combined["weight"] == 334]


# **Observation:**
# 
# - The maximum number of 334 connections is shown below to be between **Robert Baratheon and Eddard Stark**, who were pivotal co-characters in the first book.

# ## **Creating a Graph Network (for each book as well as all the books combined)**

# In[11]:


# nx.from_pandas_edgelist returns a graph from a Pandas DataFrame containing an edge list
G1 = nx.from_pandas_edgelist(book1, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())
G11=  nx.from_pandas_edgelist(book1, 'Person 1', "Person 2")

G2 = nx.from_pandas_edgelist(book2, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())
G22 =nx.from_pandas_edgelist(book2, 'Person 1', "Person 2", edge_attr = "weight")

G3 = nx.from_pandas_edgelist(book3, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G4 = nx.from_pandas_edgelist(book4, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G5 = nx.from_pandas_edgelist(book5, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())

G = nx.from_pandas_edgelist(books_combined, 'Person 1', "Person 2", edge_attr = "weight", create_using = nx.Graph())
GG = nx.from_pandas_edgelist(books_combined, 'Person 1', "Person 2", create_using = nx.Graph())


# In[12]:


print(G)
print(GG)


# ### **Number of nodes and edges across all books**

# In[13]:


nx.info(G)


# ### **Creating functions to calculate the number of unique connections per character, Degree Centrality, Eigenvector Centrality, and Betweenness Centrality**

# In[14]:


# The number of unique connections

def numUniqueConnec(G):
    numUniqueConnection = list(G.degree())
    
    numUniqueConnection = sorted(numUniqueConnection, key = lambda x:x[1], reverse = True)
    
    numUniqueConnection = pd.DataFrame.from_dict(numUniqueConnection)
    
    numUniqueConnection.columns = (["Character", "NumberOfUniqueHCPConnections"])
    
    return numUniqueConnection


# In[15]:


numUniqueConnec(G)


# **Observation:**
# 
# - **Tyrion Lannister** is the character with the **highest number of unique connections**, followed by Jon Snow and Jaime Lannister.

# In[16]:


# Degree Centrality 
''' nx.degree_centrality(G) computes the degree centrality for nodes.
The degree centrality for a node v is the fraction of nodes it is connected to.'''

def deg_central(G):
    deg_centrality = nx.degree_centrality(G)
    
    deg_centrality_sort = sorted(deg_centrality.items(), key = lambda x:x[1], reverse = True) #sort the degree centralities of characters in decending order
    
    deg_centrality_sort = pd.DataFrame.from_dict(deg_centrality_sort)
    
    deg_centrality_sort.columns = (["Character", "Degree Centrality"])
    
    return deg_centrality_sort


# In[17]:


deg_centrality_sort = deg_central(G)
deg_central(G)


# **Observation:**
# 
# - **Tyrion Lannister is the character with the highest Degree Centrality**, followed by Jon Snow and Jaime Lannister.
# 
# The higher the number of connections, the higher the Degree Centrality.

# In[18]:


# Eigenvector Centrality
''' nx.eigenvector_centrality computes the eigenvector centrality for the graph G.
Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors. 
The eigenvector centrality for node i is the i-th element of the vector x defined by the equation Ax=kx'''

def eigen_central(G):
    eigen_centrality = nx.eigenvector_centrality(G, weight = "weight")
    
    eigen_centrality_sort = sorted(eigen_centrality.items(), key = lambda x:x[1], reverse = True)
    
    eigen_centrality_sort = pd.DataFrame.from_dict(eigen_centrality_sort)
    
    eigen_centrality_sort.columns = (["Character", "EigenVector Centrality"])
    
    return eigen_centrality_sort


# In[19]:


eigen_central(G)


# **Observation:**
# 
# - **Tyrion Lannister** is also the leader when it comes to **Eigenvector Centrality**, followed by Cersei Lannister and Joffrey Baratheon.

# In[20]:


# Betweenness Centrality
'''nx.betweenness_centrality(G) computes the shortest-path betweenness centrality for nodes.
Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v. 
'''
def betweenness_central(G):
    betweenness_centrality = nx.betweenness_centrality(G, weight = "weight")
    
    betweenness_centrality_sort = sorted(betweenness_centrality.items(), key = lambda x:x[1], reverse = True)
    
    betweenness_centrality_sort = pd.DataFrame.from_dict(betweenness_centrality_sort)
    
    betweenness_centrality_sort.columns = (["Character", "Betweenness Centrality"])
    
    return betweenness_centrality_sort


# In[21]:


betweenness_central(G)


# However, for the case of **Betweenness Centrality**, it is **Jon Snow** who's at the top. 
# 
# So, Jon Snow is the central character that seems to best connect different, disparate groupings of characters.
# 
# 

# ## **Visualizing Graph Networks using Plotly** 
# 
# 

# In[22]:


def draw_plotly_network_graph(Graph_obj, filter = None, filter_nodesbydegree = None):
    G_dup = Graph_obj.copy()

    degrees = nx.classes.degree(G_dup)
    
    degree_df = pd.DataFrame(degrees)
    
    # Filter out the nodes with fewer connections
    if filter is not None:
        top = deg_centrality_sort[:filter_nodesbydegree]["Character"].values# sort the top characters using filter_nodesbydegree
        
        G_dup.remove_nodes_from([node
                             for node in G_dup.nodes
                             if node not in top
                            ]) 

    pos = nx.spring_layout(G_dup)

    for n, p in pos.items():
        G_dup.nodes[n]['pos'] = p

    # Create edges 
    # Add edges as disconnected lines in a single trace and nodes as a scatter trace
    edge_trace = go.Scatter(
        x = [],
        y = [],
        line = dict(width = 0.5, color = '#888'),
        hoverinfo = 'none',
        mode = 'lines')

    for edge in G_dup.edges():
        x0, y0 = G_dup.nodes[edge[0]]['pos']
        
        x1, y1 = G_dup.nodes[edge[1]]['pos']
        
        edge_trace['x'] += tuple([x0, x1, None])
        
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x = [],
        y = [],
        text = [],
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            showscale = True,
            colorscale = 'RdBu',
            reversescale = True,
            color = [],
            size = 15,
            colorbar = dict(
                thickness = 10,
                title = 'Node Connections',
                xanchor = 'left',
                titleside = 'right'
            ),
            line = dict(width = 0)))

    for node in G_dup.nodes():
        x, y = G_dup.nodes[node]['pos']
        
        node_trace['x'] += tuple([x])
        
        node_trace['y'] += tuple([y])

    # Color node points by the number of connections
    for node, adjacencies in enumerate(G_dup.adjacency()):
        node_trace['marker']['color'] += tuple([int(degree_df[degree_df[0] == adjacencies[0]][1].values)])
        
        node_info = adjacencies[0] + '<br /># of connections: ' + str(int(degree_df[degree_df[0] == adjacencies[0]][1].values))
        
        node_trace['text'] += tuple([node_info])

    # Create a network graph
    fig = go.Figure(data = [edge_trace, node_trace],
                 layout = go.Layout(
                    title = '<br>GOT network connections',
                    titlefont = dict(size = 20),
                    showlegend = False,
                    hovermode = 'closest',
                    margin = dict(b = 20, l = 5, r = 5, t = 0),
                    annotations=[ dict(
                        text = "",
                        showarrow = False,
                        xref = "paper", yref = "paper") ],
                    xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
                    yaxis = dict(showgrid = False, zeroline = False, showticklabels = False)))

    iplot(fig)


# **Drawing the first graph for all the data**

# In[23]:


draw_plotly_network_graph(Graph_obj = G, filter = None, filter_nodesbydegree = None)


# **This seems like a very complicated graph. So, let's draw a graph of only the top 50 characters across all the books.**

# ### **All Books Combined**

# In[24]:


draw_plotly_network_graph(Graph_obj = G, filter = "Yes", filter_nodesbydegree = 50)


# **Observation:**
# 
# - **Tyrion Lannister** is the most connected character across the book series, followed by Jon Snow and Jamie Lannister.
# 
# Now let's visualize this for the individual books as well:

# ### **Book 1**

# In[25]:


draw_plotly_network_graph(Graph_obj = G1, filter = "Yes", filter_nodesbydegree = 50) #Top 50 characters network in Book 1


# #### **Summary - Book 1**
# 
# 1. **Eddard Stark** is the most connected character, followed by **Robert Baratheon**. 
# 2. Tyrion, Catelyn, and Jon are in the top 5 characters.
# 3. Rob, Sansa, and Bran are all well-connected too, but the first book mostly revolves around Ed Stark and Robert Baratheon.
# 4. Cersei Lannister, Joffrey Baratheon, Jamie Lannister, Arya Stark, Daenerys, and Drogo are the other well-connected characters in this book.
# 
# The above findings make sense considering the plot of Book 1. Robert Baratheon, the king of the seven kingdoms, visits the House of Stark to offer Eddard Stark the position of Hand of the King, which Stark accepts. Eddard Stark's two daughters Arya and Sansa, also accompany him to the King's Landing, while his son Robb Stark looks after the House of Stark in Eddard's absence. The book eventually ends with the death of Robert Baratheon and the execution of Ed Stark by the new king Joffrey Baratheon. Robert and Eddard's importance in the story and their links to other characters in the book makes it logical that they are the two most connected characters in Book 1 of the series, with the **highest Degree Centrality measures** as seen in the table below. Tyrion Lannister, already the next most important character in Book 1, gains prominence in the coming books and becomes the most connected character in the book series overall.

# ### **Book 2**

# In[26]:


draw_plotly_network_graph(Graph_obj = G2, filter = "Yes", filter_nodesbydegree = 50)


# #### **Summary - Book 2**
# 1. **Tyrion Lannister** has become the central character, followed by **Joffrey Baratheon** and **Cersei Lannister**. 
# 2. **Arya Stark has started gaining prominence** with her being connected to Bran and Robb Stark.
# 3. Catelyn Stark has been pushed down from the top 5, but Robb Stark and Theon Greyjoy have gained importance.
# 4. Robert Baratheon and Eddard Stark have lost a huge amount of importance because they both died at the end of the first book. 
# 
# The list of top 20 characters in the table below by Degree Centrality lends credence to the above insights.
# 
# The findings above make sense from the plot of Book 2. **Tyrion Lannister**, the new King's Hand, is the most central character with the **highest Degree Centrality** measure. **Joffrey Baratheon**, who became king after Robert's death, has become a prominent character next to Tyrion. Even though Joffrey is king, **Cersei Lannister,** his mother, makes her own decisions which get implemented through Joffrey. Also, thousands of years ago, a huge wall was constructed with ice to defend the kingdoms from the rare creatures in the north. This wall was defended and maintained by the brotherhood of the Night's Watch, of which Jon Snow starts becoming an important member. The death of Eddard Stark also brings more unity to the north. Robb Stark and his mother Catelyn Stark make allies to take revenge for their father's death, and this is the reason for their increased occurrence (and hence importance) in Book 2.

# In[27]:


deg_central(G2)[:20]


# ### **Book 3**

# In[28]:


draw_plotly_network_graph(Graph_obj = G3, filter = "Yes", filter_nodesbydegree = 50)


# #### **Summary - Book 3**
# 1. **Tyrion Lannister** remains the most central character, followed by **Jon Snow** & **Joffrey Baratheon**. 
# 2. **Jon Snow has risen multiple places and is one of the most connected characters in Book 3**, second only to Tyrion Lannister.  
# 3. Sansa Stark & Jaime Lannister have also gained prominence.
# 4. Robb Stark is also in the top 5 most connected characters.
# 
# The above findings make sense considering the plot of Book 3. With **Joffrey Baratheon** being king and **Tyrion Lannister** being the King's hand, these two characters are very central to the story with a high **Degree Centrality** as shown in the table below.  **Jon Snow** also becomes one of the most central characters in the story as he builds good relations with the wildlings, and also falls in love with one of them. Robb Stark and Catelyn Stark make allies to avenge Eddard Stark's death, and are invited to the red wedding, but are both murdered, and their prominence in the story also explains their high Degree Centrality ranking.

# In[29]:


deg_central(G3)[:20]


# ### **Book 4**

# In[30]:


draw_plotly_network_graph(Graph_obj = G4, filter = "Yes", filter_nodesbydegree = 50)


# #### **Summary - Book 4**
# 
# 1. An interesting insight here is that the most connected character in Book 4 is **Jaime Lannister** followed by **Cersei**.
# 2. Brienne and Tyrion Lannister follow them but are way below them in terms of actual connections and Degree Centrality values.
# 3. Arya Stark is no longer in the top 10.
# 
# In the plot of Book 4, **Jaime Lannister** is the most centralized character as shown in the **Degree Centrality** table below, but **Stannis Baratheon** acted as a bridge between different communities, which makes his character have more control over the network. That can be seen in the **Betweenness Centrality** table below.

# In[31]:


deg_central(G4)[:20]


# In[33]:


betweenness_central(G4)[:20]


# ### **Book 5**

# In[32]:


draw_plotly_network_graph(Graph_obj = G5, filter = "Yes", filter_nodesbydegree = 50)


# #### **Summary - Book 5**
# 
# 1. As expected, **Jon Snow** and **Daenerys** are the most connected characters in this book.
# 2. Stannis, Tyrion, and Theon Greyjoy follow them.
# 3. If you look closely, Stannis Baratheon (orange node in the middle) seems to be connecting multiple groups, i.e., he has high Betweenness Centrality.
# 
# **Jon Snow, Daenerys, and Stannis** were the most centralized characters in this book as they have connections with people from different communities, as we see from both the **Degree Centrality** and **Betweenness Centrality** tables below. Daenerys is under attack, but she marries Hizdahr zo Loraq to end the violence and to make allies. Daenerys' dragon appears in the fighting pits of Meereen. Tyrion was captured by Jorah Mormont, who was one of the commanders of Daenerys' army. Cersei plans to have Margaery Tyrell arrested for her son's murder, but she gets arrested herself. Arya becomes an acolyte at the House of Black and White, where she is trained as an assassin. After establishing a truce with the wildlings, Jon Snow is stabbed by men of the Night's Watch.

# In[33]:


deg_central(G5)[:20]


# In[34]:


betweenness_central(G5)[:20]


# A higher betweenness centrality means that the node is crucial for the structure of the network, and Stannis Baratheon seems to have the characterstics to be the holding the network together.

# ## **Evolution of central characters through the books**

# In[35]:


# Creating a list of degree centrality of all the books
Books_Graph = [G1, G2, G3, G4, G5]

evol = [nx.degree_centrality(Graph) for Graph in Books_Graph]


degree_evol_df = pd.DataFrame.from_records(evol)

degree_evol_df.index = degree_evol_df.index + 1

# Plotting the degree centrality evolution of few important characters
fig = px.line(degree_evol_df[['Eddard-Stark', 'Tyrion-Lannister', 'Jon-Snow', 'Jaime-Lannister', 'Cersei-Lannister', 'Sansa-Stark', 'Arya-Stark']],
             title = "Evolution of Different Characters", width = 900, height = 600)

fig.update_layout(xaxis_title = 'Book Number',
                   yaxis_title = 'Degree Centrality Score',
                 legend = {'title_text': ''})

fig.show()


# ### **Summary**
# 1. **Eddard Stark was the most popular character in Book 1**, but he was killed at the end of the book.
# 2. Overall, from all five books, **Tyrion Lannister is the most popular character in the series.**
# 3. There is a **sudden increase in Jon Snow's popularity in Book 5**.
# 4. **Jaime and Cersei Lannister** remain central characters throughout.
# 5. Sansa & Arya's importance is high in the first few books, but it decreases thereafter.

# ## **Community Detection**

# In[36]:


import community as community_louvain
import matplotlib.cm as cm
import colorlover as cl


# In[37]:


# compute the best partition
partition = community_louvain.best_partition(G, random_state = 12345)

partition_df = pd.DataFrame([partition]).T.reset_index()

partition_df.columns = ["Character", "Community"]

partition_df


# In[38]:


partition_df["Community"].value_counts().sort_values(ascending = False)


# In[39]:


colors = cl.scales['12']['qual']['Paired']

def scatter_nodes(G, pos, labels = None, color = 'rgb(152, 0, 0)', size = 8, opacity = 1):
   

    trace = go.Scatter(x = [], 
                    y = [],  
                    text = [],   
                    mode = 'markers', 
                    hoverinfo = 'text',
                           marker = dict(
            showscale = False,
            colorscale = 'RdBu',
            reversescale = True,
            color = [],
            size = 15,
            colorbar = dict(
                thickness = 10,
                xanchor = 'left',
                titleside = 'right'
            ),
            line = dict(width = 0)))
    
    for nd in G.nodes():
        x, y = G.nodes[nd]['pos']
        
        trace['x'] += tuple([x])
        
        trace['y'] += tuple([y])
        
        color = colors[partition[nd] % len(colors)]
        
        trace['marker']['color'] += tuple([color])
        
    for node, adjacencies in enumerate(G.adjacency()):
        node_info = adjacencies[0]
        
        trace['text'] += tuple([node_info])

    return trace    

def scatter_edges(G, pos, line_color = '#a3a3c2', line_width = 1, opacity = .2):
    trace = go.Scatter(x = [], 
                    y = [], 
                    mode = 'lines'
                   )
    
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        
        x1, y1 = G.nodes[edge[1]]['pos']
        
        trace['x'] += tuple([x0, x1, None])
        
        trace['y'] += tuple([y0, y1, None])
        
        trace['hoverinfo'] = 'none'
        
        trace['line']['width'] = line_width
        
        if line_color is not None:                 # when line_color is None, a default Plotly color is used
            trace['line']['color'] = line_color
    
    return trace


# In[41]:


def visualize_community(Graph, filter = "Yes", filter_nodes = 100):
    G_dup = G.copy()

    degrees = nx.classes.degree(G_dup)
    
    degree_df = pd.DataFrame(degrees)
    
    if filter is not None:
        top = deg_centrality_sort[:filter_nodes]["Character"].values
        
        G_dup.remove_nodes_from([node
                             for node in G_dup.nodes
                             if node not in top
                            ])

    pos = nx.spring_layout(G_dup, seed = 1234567)

    for n, p in pos.items():
        G_dup.nodes[n]['pos'] = p

    trace1 = scatter_edges(G_dup, pos, line_width = 0.25)
    trace2 = scatter_nodes(G_dup, pos)
    
    fig = go.Figure(data = [trace1, trace2],
             layout = go.Layout(
                title = '<br> GOT Community Detection',
                titlefont = dict(size = 20),
                showlegend = False,
                hovermode = 'closest',
                margin = dict(b = 20, l = 5, r = 5, t = 40),
                annotations = [ dict(
                    text = "",
                    showarrow = False,
                    xref = "paper", yref = "paper") ],
                xaxis = dict(showgrid = False, zeroline = False, showticklabels = False),
                yaxis = dict(showgrid = False, zeroline = False, showticklabels = False)))
    
    iplot(fig)


# In[42]:


visualize_community(Graph = G, filter = "Yes", filter_nodes = 100)


# ### **Summary**
# 1. The Louvain method was able to find **14 different communities**.
# 2. Here are some descriptions of a couple of communities:
#     - **The yellow nodes represent Dothraki** consisting of Drogo, Danaerys, Nahaaris, etc.
#     - **The light purple nodes** represent Tyrion, Cersei, Tywin, Joffrey, Sansa, etc.
#     - **The red nodes** consist of Robb, Catelyn, Brienne, Jaime, etc.
#     - Arya is coupled with Gendry and Beric-Dondarrion in the **orange colored nodes.**
#     - **The light blue** nodes represent another very important community consisting of the **Night's Watch**, including Jon Snow, Jeor-Mormont, Samwell-Tarly, Gilly, Bowen-Marsh, etc.
#     - Similar inferences can be made for other nodes as well.
