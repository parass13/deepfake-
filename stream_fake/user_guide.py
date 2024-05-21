import streamlit as st


def show_user_guide():
    st.header(' ')
    st.header(' ')
    st.header("User guide")
    st.header(' ')

    container = st.container(border=True)
    container.markdown(''':blue[**Step 1:**]''')
    container.write("Create an account to get access")

    container2 = st.container(border=True)
    container2.markdown(''':blue[**Step 2:**]''')
    container2.write("Upload your Images")

    container3 = st.container(border=True)
    container3.markdown(''':blue[**Step 3:**]''')
    container3.write("Get the results")

    #row1 = st.columns(1)
    #row2 = st.columns(1)
    #row3 = st.columns(1)


    #for col in row1:
    #with st.container():
            #tile = col.container(height=120)
    #st.title(":balloon:")

    #st.subheader("Step 1: ")
    #for col in row1:
        #tile = col.container(height=120)
        #st.markdown(":user: **Step 1:**")
        #st.write("Create an account and subscribe to get access")
    
    #tile.title(":balloon:")
    #st.write("This is outside the container")

    #Now insert some more in the container
    #container.write("This is inside too")
    
