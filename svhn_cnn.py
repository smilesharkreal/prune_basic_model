import scipy.io as sio

traindata = sio.loadmat('svhndata/train_32x32.mat')
testdata = sio.loadmat('svhndata/test_32x32.mat')

print("Train Data Shape:", traindata['X'].shape)
print("Test Data Shape:", traindata['y'].shape)

