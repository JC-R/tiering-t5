#!usr/bin/env python

from tiering.T5Processor import T5Processor

ARTICLE = """ There are some misconceptions and confusion about what data scientists do day-to-day. There seems to be not a clear understanding of what they deliver, how much coding they do, who they work with, etc. This is not without cause of course. What a data scientist does day-to-day depends a lot on the company, team structure, person or even sometimes on the industry they work in. But I believe there are some common things every data scientist goes through in their daily work.
In this article, I will share hour by hour what a data scientist does to give you a better idea about what this job entails. For context, I will talk about someone who works in the data science team of a company. We will start from getting assigned to a new project to delivering a project. Of course, a typical project lasts a lot longer than a couple of days, but let me give you a condensed timeline for the sake of the example.
Day 1
9 am
Your supervisor approaches you to talk about a new project. Team A from your company needs something done with their data. But your supervisor also doesn’t really know what they need because they weren’t able to explain it very well to him/her. You tell your supervisor you will get in contact with them.
You send an email or call the person who requested the project and set up a meeting with them and some other people from their team.
10 am
You go to the meeting accompanied by your supervisor and try to get a feeling of what they want. It turns out they heard from some other team that you predicted their xyz and it was really useful. They ask you if you can predict their abc which is very similar to xyz. They think that it shouldn’t take you so long since you can use the same model.
You explain to them that even though xyz and abc are really similar, you need to set up a whole new system using their data. You tell them that this project needs some work.
They say they will send you a sample amount of data. You agree to look into their data and have another meeting with them to discuss the problem further.
11 am
You receive a small amount of data and familiarize yourself with it. You take notes and write questions to make the next meeting more efficient.
1 pm
You meet with team A in order to understand what they need better. You listen to their take on the problem. They use a lot of jargon and keywords special to their area. You ask for clarifications every time you don’t understand something.
You also ask the questions you prepared while studying the data. The more you talk with them, the more you become aware that what they need is very different from what they think they do.
You explain to this to them and talk about why what they need is different and much simpler.
You explain the capabilities and limitations of machine learning and even a little bit about how a machine learning model works.
As a result of this discussion, you agree on a tentative goal for the project. You inform them that you need to first understand the problem better and then you can revise the goal if needed.
2 pm
You get in touch with some people who have more in-depth knowledge of how team A’s system works and also someone else who knows how team A’s data is collected. You interview them to learn about the domain, the data collection process and the exceptions if there are any.
3 pm
You come up with a plan for how to approach the project. You make sure to consider AI ethics implication of the project that you’re about to start and note them down to discuss with your stakeholders if any.
4 pm
You talk to some other data scientists in your team, explain your approach to them and get their feedback.
You revise your plan and the goal based on your conversations with domain experts and your colleagues.
Day 2
9 am
You meet with your stakeholders to present them your revised plan. You talk about what you learned from the experts in team A’s domain, how you revised the goal accordingly, the potential ethics issues you need to look out for and the proposed approach you came up with.
You discuss the content with them and come up with a deliverable and a timeline. You also agree to meet periodically during the project to update them.
10 am
You receive all the data and start exploring it. You make visualizations and look closely into the data to find out more about:
the distribution of features
missing values
unbalance in the data
bias against a certain race, sex, gender, ethnic group or other groups
which features you can or cannot use due to data laws
If there are any problems with the data or things you cannot understand like a confusing feature name, you go back to your connection in team A who knows about the data and ask them for clarification.
11 am
You prepare your findings from the data into presentable content and either send it to your stakeholders or present it yourself. If there are new challenges or implications caused by your findings, you share it with them and discuss to realign on the goal and the approach.
1 pm
You again work on the data to figure out what other data sources you can use for the goal at hand. You do some research into other freely available datasets inside your company and other paid or free data sources available on the internet that can support this project.
Day 3
9 am
You bring all the data you have together and explore again, this time to have the data ready for training. While doing so, you:
detect and deal with outliers
find missing value points and decide on a way to substitute or get rid of them
work up a way to include your categorical features
1 pm
You think of some new features to generate. Brainstorm with some other team members about how to augment, merge or alter the features you have to help the algorithm have a better performance
Depending on the amount of quality/usable data you have, you decide on the amount of test data needed. You also decide on how to split your data as this decision depends on the type of problem.
4 pm
You spent most of the day cleaning, generating new features and having the training and testing sets ready. From now on, you’re going to focus on training models. But before you start modelling you have to determine the success factor of this project.
You decide on the evaluation metric you’re going to use based on what your stakeholders need. You talk to them again to make sure you are optimizing the results to what they need. You explain to your stakeholders that this is important because if you use the wrong evaluation metric, you will be optimizing the model for the wrong type of performance. And with the wrong type of performance, even if the model performs very well, it might end up being unusable for your stakeholders.
Day 3
9 am
You decide on a couple of models that are appropriate for this type of problem.
You build a first model and evaluate the performance using the evaluation metric or metrics you determined.
2 pm
You try different settings and hyperparameter values to improve your model’s performance.
Day 4
9 am
After implementing and training a couple of different models you choose the most promising ones. In order to achieve higher performance, you try creating some new features and excluding some features.
Depending on the model you’re using, you can check the helpfulness of features to see which one helps the model predict better.
1 pm
After you attain an acceptable level of performance you finalize the documentation about your code and the model that you’d been writing on the side during the development process.
You prepare content to present your results in an understandable way.
3 pm
You present your results to your stakeholders mentioning what you found out about their system and what else can be improved. You agree to have the model ready for use in a certain time.
Day 5
9 am
You get in contact with someone who is going to use the output of your model. They explain in more detail the format they need the results in.
10 am
You meet with a dev/ops engineer to help you deploy your model. You tell him/her about the problem, your model, how you need the data to come in and what comes out of your model. Together, you make a plan for deployment.
1 pm
You finalise your code and documentation and send it to the dev/ops engineer. While he’s getting ready to integrate it into your company’s production, you keep in close contact with him to answer questions and solve arising problems.
2 pm
At the same time, you send information to your customer team A about how they can start using your model once it’s deployed.
5 pm
You talk about your week’s work with your colleagues while wrapping up the week. You feel proud of yourself for being such a kick-ass data scientist.
You accept the compliments on your hard work.
That’s it! You had a busy week but it’s all worth it when you see the results of what you’ve built!
I hope you enjoyed this little story of a week in a data scientist’s life. Keep in mind that this is not a perfect description of what a data scientist does, it’s merely a combination of my experience and stories I’ve heard from my podcast guests or colleagues. Depending on the company, a data scientist might have more or fewer responsibilities. After all, every company is unique.
There are things I know is part of a data scientist’s responsibilities, for example, maintaining previously built models, weekly team meetings, periodic meetings with your stakeholders to keep them up-to-date on your progress, how long it takes to actually arrange a meeting with people in big organisations and how long it actually takes to get your model deployed.
Of course, you might also be aiming for a freelancer position and needless to say, that sort of data science work would look very different.
My goal with this article was to give you a small peek inside the world of data science. Hope I achieved it! Let me know what you think about it in the comments.
        """


t5 = T5Processor(model="t5-base")
t5.fit(ARTICLE, split=True)
e = t5.doc_embeddings()
print(e)


