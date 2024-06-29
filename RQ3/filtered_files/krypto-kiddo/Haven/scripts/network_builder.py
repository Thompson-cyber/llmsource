# A deprecated script that was planned for scraping data and generating the network graph, built by ChatGPT
# This code was never used in the main project for miscellaneous reasons
# There's nothing really interesting to see here. 




## CODE PART 1 , generated by ChatGPT ##

import requests

# Set the API endpoint URLs
block_details_url = "https://rpc.nearprotocol.com/block_details?block_id={}"
transaction_details_url = "https://rpc.nearprotocol.com/tx/{transaction_id}/results"

# Define a function to retrieve transaction details
def get_transaction_details(transaction_id):
    response = requests.get(transaction_details_url.format(transaction_id))
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Define a function to construct the network
def construct_network(start_block_id, end_block_id):
    # Initialize an empty graph
    G = nx.Graph()

    # Loop through the blocks
    for block_id in range(start_block_id, end_block_id):
        # Retrieve the block details
        response = requests.get(block_details_url.format(block_id))
        if response.status_code == 200:
            block_details = response.json()
            transactions = block_details['result']['transactions']
            # Loop through the transactions
            for transaction in transactions:
                # Retrieve the transaction details
                transaction_id = transaction['hash']
                transaction_details = get_transaction_details(transaction_id)
                if transaction_details:
                    # Add the sender and receiver addresses to the graph
                    sender = transaction_details['receipts'][0]['predecessor_id']
                    receiver = transaction_details['actions'][0]['args']['receiver_id']
                    G.add_node(sender)
                    G.add_node(receiver)
                    # Add an edge between the sender and receiver addresses
                    G.add_edge(sender, receiver)
    return G

## END OF PART 1 ##

## PART 2 : DRIVER CODE , generated by ChatGPT ##

import networkx as nx
import matplotlib.pyplot as plt

# Define the start and end block IDs
start_block_id = 1
end_block_id = 100

# Construct the network
G = construct_network(start_block_id, end_block_id)

# Print some basic statistics about the network
print(nx.info(G))

# Draw the network
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=False, node_size=10)
plt.show()

## END OF PART 2 ## 



"""
                                 d888b
          __,.---"""-.          8888888          .-"""---.,__
      _.-' ._._._._,_ `"-.      8888888      .-"` _,_._._._. '-._
,__.-' _/_/_/_/_/_/_/_/_,_`'.    Y888P    .'`_,_\_\_\_\_\_\_\_\_'-.__,
 `'-._/_/_/_/_/_/_/_/_/_/_/,_`"._ dWb _."`_,_\_\_\_\_\_\_\_\_\_\_.-'`
      '-;-/_/_/_/_/_/_/_/_/_.--. WWWWW .;;,_\_\_\_\_\_\_\_\_\-;-'
          /_/_/_/_/_/_/_/_//  e \IIIII;;a;;;\_\_\_\_\_\_\_\_\
            '-/_/_/_/_/_/ /   ,-'IIIII'=;;;;; \_\_\_\_\_\-'
                /_/_/_/_ /   /   88888   ;;;;; _\_\_\_\
                    '-/_/|   |   88888   ;;;;;\_\-'
                          \   \  88888  ;;;;;
                           '.  '.'888'.;;;;'
                             '.  '888;;;;'
                               '. .;;;;'
                                .;;;;'.
                              .;;;;8.  '.
                            .;;;;'888'.  '.
                           .;;;;  888  \   \
                           ;;;;   888  |   |
                           ';;;;  888  /   /
                            ';;;;.888.'  .'
                              ';;;;8'  .'
                                ';'  .'
                               .'  .;;;.
                              /   /8\;;;;
                             /   /888;;;;,
                             |   |888 ;;;;
                             \   \888;;;;'
                              '.  '8;;;;'
                                '.;;;;'
                         jgs    ;;;;` \
                               ;;;;8|  |
                               ';;;8/  /
                                '-'8'-'
                                   8
"""