#   Function to obtain magnetization value through for loop

def magnetization_sum(nlayers,iterator,iterator2,start_matrix):
    magnetization_sumer=0
    for k in range(0,nlayers):                          #   Depth
        for i in range(1,iterator-1):                   #   Rows
            for j in range(1,iterator2-1):              #   Columns
                magnetization_sumer=magnetization_sumer+start_matrix[k,i,j]
    print("For Loop",magnetization_sumer)

#   End of function