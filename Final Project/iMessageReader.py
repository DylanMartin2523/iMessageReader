import sqlite3
import sys
from wordcloud import WordCloud, STOPWORDS
import collections
import datetime
import matplotlib.pyplot as plt
from time import strftime
from sklearn.cluster import KMeans
import pandas as pd


# Code written for project
def analyze_data(df_messages, verbose, sample_size):
    # Builds dictionary of all phone numbers and the number of texts corresponding to each phone number
    number_time = {}
    for id in df_messages["phone_number"]:
        if not (str(id) in 'nan'):
            try:
                number_time[id] = number_time[id] + 1
            except KeyError:
                number_time[id] = 0
    # Sets up data frame that is going to be used for the k-means analysis
    coordinate_df = pd.DataFrame.from_dict(number_time, orient='index')
    if verbose == 'y':
        coordinate_df = coordinate_df.sample(sample_size)

    # This for loops calculates the average difference for every text by every number in the coordinate_df data frame.
    # The average difference is added to the dictionary avgDiff along with the phone number as a key.
    avgDiff = {}
    for id in list(coordinate_df.index):
        forAvg = []
        for x in df_messages[df_messages == id]['phone_number'].dropna().index:
            date = df_messages['date'][x]
            if df_messages['date'][x] in forAvg:
                continue
            elif df_messages['phone_number'][x] == id:
                forAvg.append(date)
        if len(forAvg) == 1:
            avgDiff[id] = 0
            continue
        for x in range(0, len(forAvg) - 1):
            forAvg[x] = abs(forAvg[x] - forAvg[x + 1])
        forAvg.remove(forAvg[len(forAvg) - 1])
        avgDiff[id] = sum(forAvg) / len(forAvg)
    # Sort data so that the indices in avgDiff_df and coordinate_df match up
    avgDiff_df = pd.DataFrame.from_dict(avgDiff, orient='index')
    avgDiff_df = avgDiff_df.sort_index()
    # Changes the difference from avg num of nanoseconds between texts to avg num of days between texts
    avgDiff_df[0] = avgDiff_df[0].map(lambda x: x / 8.64E+13)
    coordinate_df = coordinate_df.sort_index()

    # Make a new column that adds the average difference between texts to the coordinate_df and renames columns
    coordinate_df[1] = avgDiff_df[0]
    coordinate_df.columns = ['Number of Texts', 'Average Days Between Texts']
    # K-means analysis
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(coordinate_df)
    # Making plot
    plt.scatter(coordinate_df['Number of Texts'], coordinate_df['Average Days Between Texts'], c=kmeans.labels_, cmap='rainbow')
    print(kmeans.labels_)
    coordinate_df['Clusters'] = pd.Series(kmeans.labels_, index=coordinate_df.index)
    plt.title('Number of Texts Compared to Average Days Between Texts')
    plt.xlabel('Number of Texts')
    plt.ylabel('Average Days Between Texts')
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.show()

    print(coordinate_df)

# Extra stuff I had previously included when playing around with this data
def extras(df_messages):
    dt = datetime.date(year=2001, day=1, month=1)
    dtu = (dt - datetime.date(1970, 1, 1)).total_seconds()
    df_messages['date'] = df_messages['date'].map(
        lambda date: datetime.datetime.fromtimestamp(int(date / 1000000000) + dtu))

    months = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

    for date in df_messages['date']:
        months[date.month] = months[date.month] + 1
    list1 = sorted(months.items())
    x, y = zip(*list1)
    plt.title("Texts Per Month")
    plt.xlabel('Months')
    plt.ylabel("Texts")
    plt.bar(x, y, edgecolor='black')
    plt.show()

    years = {2017: 0, 2018: 0, 2019: 0}

    for date in df_messages['date']:
        years[date.year] = years[date.year] + 1

    list1 = sorted(years.items())
    x, y = zip(*list1)
    plt.title("Texts Per Year")
    plt.xlabel('Years')
    plt.ylabel("Texts")
    plt.bar(x, y, edgecolor='black')
    plt.show()

    rw = {'Sent': 0, 'Recieved': 0}
    for x in rw:
        for i in df_messages['is_sent']:
            if i == 1:
                rw['Sent'] = rw['Sent'] + 1
            else:
                rw['Recieved'] = rw['Recieved'] + 1

    list1 = sorted(rw.items())
    x, y = zip(*list1)
    plt.title("Sent / Recieved")
    plt.xlabel('Status')
    plt.ylabel("Texts")
    plt.bar(x, y, edgecolor='black')
    plt.show()



if __name__ == '__main__':
    if len(sys.argv) < 1:
        sys.exit("USAGE: " + sys.argv[0] + " path/to/chat.db")
    file_name = sys.argv[1]

    print(file_name)
    print("Welcome to the iMessage database analyzer")
    print("Manually set sample size? (May drastically impact speed if not used) (y/n)")
    verbose = input()
    sample_size = ""
    if verbose == 'y':
        print("Entered desired sample size")
        sample_size = int(input())
        print("Running...")
    else:
        print("Running...")

    # Code to clean up data for ease of analysis
    conn = sqlite3.connect(file_name)
    # connect to the database
    cur = conn.cursor()
    # get the names of the tables in the database
    cur.execute(" select name from sqlite_master where type = 'table' ")

    # get the 10 entries of the message table using pandas
    messages = pd.read_sql_query("select * from message", conn)

    # get the handles to apple-id mapping table
    handles = pd.read_sql_query("select * from handle", conn)
    # and join to the messages, on handle_id
    messages.rename(columns={'ROWID': 'message_id'}, inplace=True)
    handles.rename(columns={'id': 'phone_number', 'ROWID': 'handle_id'}, inplace=True)
    merge_level_1 = temp = pd.merge(messages[['text', 'handle_id', 'date', 'is_sent', 'message_id']],
                                    handles[['handle_id', 'phone_number']], on='handle_id', how='left')
    # get the chat to message mapping
    chat_message_joins = pd.read_sql_query("select * from chat_message_join", conn)
    # and join back to the merge_level_1 table
    df_messages = pd.merge(merge_level_1, chat_message_joins[['chat_id', 'message_id']], on='message_id', how='left')

    analyze_data(df_messages, verbose, sample_size)

    print("Would you like to view some extras? (y/n)")
    ans = input()
    if ans == 'y':
        extras(df_messages)
