# %%
import numpy as np
import matplotlib.pyplot as plt
import sk_dsp_comm.sigsys as ss
import matplotlib.patches as mpatches
# %%

fs, src = ss.from_wav('newSound.wav')

duration = 6

totalSample = duration * fs

frequency = 100

waveFs = fs/frequency

volume = 0.9

# sin wave to train nn
sampleWave = 0.9 * (volume * np.sin(2 * np.pi *
                                    np.arange(totalSample) * (1/waveFs))).astype(np.float32)


end = 0 + totalSample

originalSound = src[0:end]


# %%
noiseSource = 1.12 * (volume * np.sin(np.random.rand() * np.pi * (2 * np.pi/3) *
                                      np.arange(totalSample) * (1/waveFs))).astype(np.float32)


noisySound = originalSound + noiseSource

plt.title(label="original sound")

plt.plot(originalSound, color='orange')
plt.show()
plt.title(label="noise sound")
plt.plot(originalSound, color='orange')
plt.plot(noisySound, color=(0, 0, 1, 0.5))
plt.show()

ss.to_wav('sound_in.wav', fs, noisySound)


# %%

r = 32

trainData = np.array(sampleWave)
data = []
target = []

for i in range(trainData.shape[0]):
    listData = []
    for j in range(r-1):
        listData.append(trainData[(i+j)%trainData.shape[0]])
    data.append(listData)
data = np.array(data)
target = np.array(noiseSource)


# %%
def updateWeight(weight_vec, b, err_val, input_vec, lr):
    wlen = len(weight_vec)
    change = 2.0*lr*err_val
    for i in range(wlen):
        weight_vec[i] += change*input_vec[i]
    # The weight update rule for the bias
    b += change
    return weight_vec, b


# %%
errors = []
regenerateSound = []
lr = 0.01

def train():

 
    b = 1   
    weight_vec = np.random.sample(r-1)

    # weight_vec = np.append(weight_vec,b)
    for epoch in range(target.shape[0]):
        p = np.array(data[epoch]).reshape(-1, 1)

        true = target[epoch]

        predict = np.dot(weight_vec, p)
        predict += b

        error_val = true - predict
        errors.append(abs(error_val))
        regenerateSound.append(noisySound[epoch] - predict)
        weight_vec, b = updateWeight(weight_vec, b, error_val, p, lr)

    return weight_vec,b

weight_vec,b = train()

# %%
regenerateSound = np.array(regenerateSound)
plt.plot(errors[0:100])

# %%
plt.title(label="regenerate sound")
plt.plot(regenerateSound[100:])
plt.plot(originalSound, color='orange')
orange_patch = mpatches.Patch(color='orange', label='Original Signal')
blue_patch = mpatches.Patch(color='blue', label='Restored Signal')
plt.legend(handles=[orange_patch, blue_patch])


plt.show()
# %%
print()
ss.to_wav('sound_output.wav', fs, regenerateSound)

# %%


def getNewSound(path, duration):
    fs, src = ss.from_wav(path)

    totalSample = duration * fs

    frequency = 100

    waveFs = fs/frequency
    end = 0 + totalSample

    originalSound = src[0:end]
    return originalSound;

# %%
testSound = getNewSound('newTestSound.wav',6)
noiseTestSound = testSound + noiseSource

ss.to_wav('newTestSound_in.wav',fs,noiseTestSound)

regenerateTestSound = []

#%%tsts
for epoch in range(target.shape[0]):
    p = np.array(data[epoch]).reshape(-1, 1)

    true = target[epoch]

    predict = np.dot(weight_vec, p)
    predict += b

    regenerateTestSound.append (noiseTestSound[epoch] - predict)
    error_val = true - predict


    weight_vec, b = updateWeight(weight_vec, b, error_val, p, lr)
    
    
    
regenerateTestSound = np.array(regenerateTestSound)

ss.to_wav('newTestSound_out.wav',fs,regenerateTestSound)

# %%
