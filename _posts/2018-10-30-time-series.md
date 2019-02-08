---
type: posts
title:  "Event Detection in Time Series"
classes: wide
tags: [data, time series, machine learning]
toc: true
permalink: /EV_charging/
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

This project is based on data made available from [GridCure](https://www.gridcure.com/) for a Data Scientist position (which is why I am sharing the analysis but not the dataset itself). 

The dataset is a large number (N=1590) of time series which represent smart meter readings at 30 minute intervals. Each time series has 2880 data points (that is, 60 days of data taking), and labels that tell us if an EV was indeed charging in that 30-minutes time interval. The question asked is simple: 

- Given a time series of electric meter readings from a household, can we predict when an Electric Vehicles (EV) is charging? Can we predict if there is an EV at all in a given household?


Here is a sample of the first few time series, where the blue line is the meter reading, the orange line is the rolling mean over a 6 hour period, and the yellow vertical bands mark a period when a car was charging.

![Time Series](/assets/images/ev/time_series_sample.png)

As is seen in the sample above, the time series is noisy, with large spikes when a EV is charging. Depending on the household, there might also be spikes without a corresponding EV. Can we make predictions on unlabelled data? The business case for GridCure probably has something to do with grid maintenance and power output: if we can predict when EVs are hooking up to the grid, they can react accordingly.

We will take two approaches to analyze this problem. First, we can take features of the data such as rolling mean or slope, and exploit when signal and backgrounds are distributed differently. Second, we can use machine learning, namely neural networks, to analyze the data for us and pick up as much information as possible. While the first approach is more transparent, the second gives better overall results.

The techniques used in this analysis can likely be applied to lots of other problems: broadband network load, cloud throughput for computing or storage, fraud detection in banking transactions and so on.

# Analytical approach

The simplest feature is apparent in the image above: an EV charging adds a significant load on top of normal household energy consumption.

It turns out that height itself is not a great indicator, as there are too many cases in which the consumption is high even though there is no EV charging. But, as it is a transient event, the power use will go back down afterwards. Therefore we can take the rolling means: as a charge event begins, the power reading jumps above the rolling mean, which then (depending on the duration of the charge) will either level or go down when the charging is over. Therefore, two useful features will be the instantaneous difference between a point and the rolling mean, as well the slope (that is, the first derivative) of the rolling mean (we could also have taken the instantaneous slope given by point-by-point differences, but that would be too noisy).

As it turns out, these variables are actually very correlated with each other (it makes sense, the difference from mean is basically a delayed slope), so that it does not help much to consider them together. Individually, each of these variables still has some discriminating power: see below for the distributions of events as a function of the variable. Those make intuitive sense: when an EV is charging, the data is usually above the mean (here I divided by the standard deviation for each time series): this is simply due to the fact that 92% of all EV charging events are relatively short, below 10 intervals (that is, 5 hours), so the rolling mean almost never levels off.  For the slope of the mean, there is a positive increase when charging starts, which for the same reason does not go to zero before the end.

![PDF of difference from mean](/assets/images/ev/distribution_spikes_slope.png)

This is enough to make predictions as a function of a threshold, and compare those predictions to the labelled data to see how good they are. There are different metrics that measure the quality of a prediction, but in the Machine Learning community the most common one is the ROC curve: as we will see, this is not a good choice for the present dataset, and we will use instead the precision-recall curve (see [here](https://www.andybeger.com/2015/03/16/precision-recall-curves/) for a short description). Depending on the goal of the analysis and the success/fail rates that we are comfortable with, we then pick up a desired threshold to make a prediction. Given the probability distribution above, one can translate the threshold to a probability that the reading corresponds to an EV charging at any given time.

![predicted probability](/assets/images/ev/sample_pred_probability.png)

Here all intervals for which the probability is above 0.5 have been assigned as EV charging events. As we can see, it's not doing a great job: while it is picking up most obvious EV events, there are a lot of false positives! Changing the threshold to a higher value (say 0.7) would lower the false positive, but also remove a bunch of good predictions. Changing the threshold from 0 to 1 gives us the following ROC-PR curves (where we annotated the latter with probability thresholds):

![ROC curves](/assets/images/ev/ROC_PR_curves_analytic.png)


While the ROC curves are relatively good for such a low-effort approach, they are somewhat misleading. This is because our dataset is highly unbalanced: of the millions of measurements, only 2% correspond to a charging event, so it is not a big deal to correctly predict that there is no EV charging most of the time. 

When calculating the false positive rate ($$\frac{FP}{FP+TN}$$), we are swarmed by true negatives, while we would be more interested in a quantity assessing how many of our positive predictions were correct. One can use precision, defined as $$\frac{TP}{TP+FP}$$, see for example [this Wikipedia article](https://en.wikipedia.org/wiki/Precision_and_recall). The precision-recall curve (where recall is the same as the true positive rate, $$\frac{TP}{TP+FN}$$) then gives a better idea. As we can see, with the features used here it is really hard to increase the precision above 30-40%, especially if we want relatively high true positive rates (=recall, and is penalized by false negatives). This means that about more than half of our predicted positive events will instead be negatives. Depending on the business costs associated with botched predictions, this might be too much.


Finally, we can use this analysis to predict if a household has a EV at all. Practically, a EV house is one where there are more EV-like events than a normal house (for example, more spikes). We compute the fraction of time for which a household meter readings are above a certain spike threshold with respect to the rolling mean, and then choose the spike and time fraction combinations that give the best discrimination. Picking one point from the ROC curve (in this case, we have comparable EV vs. non-EV households, so that the ROC curve is a good metric), we find:   

- we can classify a household as a EV household if its energy consumption spikes above 2.4 its standard deviation for more than 1.5% of the time: this corresponds to (FPR, TPR)=(10%, 50%): with respect to a random pick at (50%,50%), we can reject 90% of non-EV households, with only 10% of them mistaken as our target.
- other choices are possible, for example one can increase the the spike threshold and decrease the time fraction threshold.

Keep these numbers in mind as we will compare the deep learning techniques to these baselines at the end of the post.

	
# Deep learning approach
Having analyzed the data with simple analytical approach, we can ask ourselves if a neural network could over-perform this treatment. As it turns out, the answer seems to be yes, and very easily. We will split the discussion in two, as we will be using very different techniques for detecting an EV charging event and classifying a household.

## Event detection

The question that we want to ask here is: what is the probability that a EV is charging at a certain moment?

Note that at the moment, the dataset is not correctly formatted to ask this question: we have multiple labels (in fact, a whole time-series of them) for each time series. Hopefully, the network should be able to learn that a charging event corresponds to a spike in energy consumption, so it makes sense for the input to be a segment of the time series of a certain length. Given that we can only give one label corresponding to a segment, it can be 0 if there was no charging at any point during that segment, and 1 otherwise. When the network is trained, it will try to predict this label.

> We could also try to teach the network what a longer charge looks like. This would be achieved by using non-binary labels, e.g. the number of charging periods during a segment. So for example with a sliding window of 5, we could have labels that go from 0 to 5. We leave this analysis for future work.

We will therefore create a new dataset with the following steps:
- For each time series of meter readings, use a sliding window to separate it into overlapping segments of a certain length (we will take $$n=5$$). We take a stride of $$m=1$$, but have also tested that the results are not that different with a stride of $$m=2$$, and the training is not much slower.
- Similarly, for the corresponding label series, we take the same sliding window and assign either 0 or 1 depending if an EV was charging at any point in that segment.
- From the labelled dataset of 1590 households and 2880 points in each time series, we now have 1590*(2880-(5-1))=4,572,840 segments. To avoid overfitting, we use segments from the first 1000 households for training, and the remaining for validation.
- After training the network, we take a rolling mean of the predictions to predict the probability at each step of the original time series. This is simply understood: let's say that the network learned that a large spike that goes on for a while corresponds to a EV charging event. Then, as the sliding window starts to overlap with that spike, the output increases and reaches a maximum when the window maximally overlaps with the spike, and then declines after passing it. A typical output is shown here:

![sliding window output of neural network](/assets/images/ev/nn_sliding_output.png)

> Note that the size of the sliding window (set at $$n=5$$) is the main factor deciding how much a short-lived spike gets spread out: for example with a sliding window of $$n=10$$ and stride $$m=1$$ going over a label series with exactly one 1 interval, the training label would be 1 for 9 overlapping segments. I found that $$n=5$$ is good for not spreading the signal too much while still letting the network learn about the temporal structure.

> In fact, using the sliding window means that the probability the network gives for a segment centered around $$T$$ is the probability of having a charging event at any moment within $$T\pm(n-1)$$. For $$n=5$$, this corresponds to a two hours interval in this dataset.

Which neural networks do we actually use? We take two simple approaches: first a logistic regression (can be coded as a one-layer fully connected network) as baseline, and a four-layer perceptron (fully connected feed-forward network with three layers with 100 neurons and an output layer). Remember that the inputs are very small segments, and that increasing the segment size lowers the localizing power of the prediction: for this reason, I did not find a 1D convolutional neural network useful.

We have already pointed out in the previous section that ROC curves are not useful for such unbalanced datasets (here, both networks easily achieve AUC>98%). Instead, we should look at the precision-recall curves below, which we also annotate with the prediction thresholds.  We also compare the neural network approach to the previous spike analysis. It can be seen that while the logistic regression is worst than our analytical approach, a relatively shallow fully connected neural network works much better (for example, doubling the AUC). Also note that, unlike with the analytical approach, we can now achieve precisions of order 60-80% without having to compromise too much on the recall/true positive rate.

![PR curve neural network](/assets/images/ev/sliding_ROC_PR_curves_nn.png)

We have also tried to pre-process the data given to the neural network: for example, instead of the raw meter readings, we have fed it the spike amplitude (the difference between current reading and the rolling mean, divided by the standard deviation of the series). This results in small deviations in the output of the network (the AUC prefers the raw data by a little, but for larger recall one can get better precision from the spike input).

For example, we can achieve precision and recall both of order 50% for a network output threshold of 0.2. Note that this is a huge improvement over the random chance line, which is much smaller due to the large imbalance between number of elements in each class. A random pick with a true positive rate of 50% would have a precision of 5% while we can increase that to 50%: this mean reducing the FP/TP ratio from 20 to 1, so whatever cost is associated with a false positive, we have reduced that by a factor of 20!


Again, for each time series we can now plot the probability that an EV is charging at any moment. We can then decide a threshold above which we classify an interval as an EV charging event. Compared to the analytical approach, we can see that the neural network output is much less noisy.

![predicted probability](/assets/images/ev/sliding_sample_pred_probability_NN.png)

## Household classifier

If we want to classify a household, we could take a similar approach as before: count the fraction of times the EV probability is above a certain threshold with our neural network approach, and choose a time fraction above a certain threshold that gives us best gains.

But, it turns out that there is much more information available in the dataset. In particular, this is a temporal dataset, with a 24-hour periodicity that should be apparent in both background and signal patterns. For example, a person commuting will usually come back home around the same time and plug their car in. We can therefore repackage the time series into a 2D matrix, that is, an image! See below for a sample of repackaged time series, where the top row shows the energy consumption, and the bottom row shows EV charging intervals in black.

![2D images](/assets/images/ev/image2D_sample.png)

As before, we can see still see the correspondence between energy consumption and EV charging. But now, instead of having to make a prediction for each point of the time series, we only have to give one prediction per household.

To make these predictions, we again try a logistic regression and a multi-layer perceptron, but this time we can also pick a 2D convolutional neural network. It does not need to be very deep, it turns out that two convolutional layers with 32 and 64 nodes followed by a dense hidden layer is sufficient. The resulting ROC curves are shown below: the convolutional neural network is vastly outperforming the other classifiers.

![ROC curves for 2D images and household classification](/assets/images/ev/images2D_ROC_PR_curves_nn.png)

This is it. We can classify if a house has an electric vehicle with over 90-95% accuracy using a simple convolutional network. `

> What is the CNN learning? There are several ways to go inquire this: I have verified that it is looking for correlations in the day-to-day direction: if we change the feature map size in the convolutional layer to be 1-dimensional (say, a $$1\times N$$ rectangle), the efficiency drops considerably. Similarly, if we split the images in shorter time intervals (say from a  sixty-day image to 10 six-days images), the efficiency again drops.

Finally, we can take a look at a sample of the ConvoNN predictions: for example, we can take the two cases (one for each class) for which it was most certain and it made the right prediction, the most uncertain case, and again two cases where it was very certain about its prediction but failed miserably (exercise: can you think of why they were misclassified?).

![2D images (mis)classified](/assets/images/ev/image2D_prediction_successes_fails.png)


# Summary

In this post, I have gone through a simple time series analysis, with the goal of event detection and client classification. I have shown both simple analytical as well as machine learning techniques, with the former giving a baseline that was greatly improved by the latter.

A simple follow-up would be to use the household classification bit to improve the event detection in the time series data, given the following point: clearly, only an EV household will have EV charging events. Given that the former is easier to classify, one could only allow the algorithm to detect charging events in EV households, therefore removing a whole lot of false positives in the no-EV time series. This would most likely greatly improve the event detection precision.

While this discussion was based on electric smart meter readings, one can imagine similar techniques in many other fields. Obviously, a lot of utilities would have similar types of problems (internet providers managing network loads, water utilities responding to spikes on top of daily routines), but other fields such as cloud computing/storage, banks fraud departments, and obviously finance, likely treat similar problems. 

This was my first foray in time series analysis, and it was fun! Feel free to let me know if this post was useful in a another field, or how you would have done things differently!


