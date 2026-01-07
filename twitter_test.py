import tweepy

bearer_token = "AAAAAAAAAAAAAAAAAAAAAK6J4QEAAAAAxWBh3xarOMRrlr8JRwL5IOwsvAo%3DAOlEYLeJh309d79hcAynSRe5T1RkoRnkBIsIJnHATPEknEVjB4"

client = tweepy.Client(bearer_token=bearer_token)

user = client.get_user(username="elonmusk")
user_id = user.data.id

response = client.get_users_tweets(id=user_id, max_results=5)

if response.data:
    for tweet in response.data:
        print(tweet.text)
else:
    print("No tweets found.")