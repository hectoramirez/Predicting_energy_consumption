# Conclusions & thoughts

<ul>
    <li> It seems to me that by lagging variables one only creates new features, <i>i.e.</i>, there is little difference in extending the dataset to (t-x) for x>>1 when predicting only the next hour (with 2 or 3 lagged features is enough). <br><br>
    <li> Shuffling the dataset gives better results despite the strong arguments not in favor of it when using LSTM networks. <a href='https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/'> Machine Learning Mastery</a> somewhere mentions that when this happens, different kind of convolutional NN should be used. I wonder if this is because of the volatile nature of this time series.<br><br>
        Furthermore, the results seem to vary considerably for different shuffled sets. Namely, it is possible that one needs to repeate the full process several times shuffling the data each time and obtain an average RMSE of the averaged one obtained from the reapeated trainnings of the LSTM network. <br><br>
    <li> Given the shape needed for the LSTM inputs, no $var(t)$ features are used (they are dropped). They might be useful if there's a way to include them. <br><br>
    <li> Usually, the model accuracy is a useful discriminator of the model performance, and the mean directional accuracy is for time series. Both are very poor in this experiment. <br><br>
    <li> The XGBoost model is not very different. This could hint that the LSTM is again not very efficient.
</ul>
<hr>
<hr>