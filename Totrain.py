from train_correntropy import train_corr
from train import train


# pre-training
train('tcnlstm', 'pre1108299.csv', 288, 7)
train('tcnlstm', 'pre1108380.csv', 288, 7)
train('tcnlstm', 'pre1108439.csv', 288, 7)
train('tcnlstm', 'pre1108599.csv', 288, 7)
train('tcnlstm', 'pre1111514.csv', 288, 7)
train('tcnlstm', 'pre1111565.csv', 288, 7)
train('tcnlstm', 'pre1114254.csv', 288, 7)
train('tcnlstm', 'pre1114515.csv', 288, 7)
train('tcnlstm', 'pre1117857.csv', 288, 7)
train('tcnlstm', 'pre1117945.csv', 288, 7)

'''
#LSTM
train('lstm', '1108299.csv', 288, 30)
train('lstm', '1108380.csv', 288, 30)
train('lstm', '1108439.csv', 288, 30)
train('lstm', '1108599.csv', 288, 30)
train('lstm', '1111514.csv', 288, 30)
train('lstm', '1111565.csv', 288, 60)
train('lstm', '1114254.csv', 288, 30)
train('lstm', '1114515.csv', 288, 30)
train('lstm', '1117857.csv', 288, 30)
train('lstm', '1117945.csv', 288, 30)
'''
'''
#TCN-LSTM-CIM
train_corr('tcnlstm', '1108299.csv', 288, 12)
train_corr('tcnlstm', '1108380.csv', 288, 12)
train_corr('tcnlstm', '1108439.csv', 288, 12)
train_corr('tcnlstm', '1108599.csv', 288, 12)
train_corr('tcnlstm', '1111514.csv', 288, 6)
train_corr('tcnlstm', '1111565.csv', 288, 6)
train_corr('tcnlstm', '1114254.csv', 288, 6)
train_corr('tcnlstm', '1114515.csv', 288, 6)
train_corr('tcnlstm', '1117857.csv', 288, 6)
train_corr('tcnlstm', '1117945.csv', 288, 6)
'''

