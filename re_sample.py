import os
import torchaudio

def re_sample(name):
        
    waveform_, sample_rate = torchaudio.load('audio_rename/'+name)

    #resample waveform to rate
    if sample_rate != rate:
        waveform_ = torchaudio.compliance.kaldi.resample_waveform(
                                            waveform=waveform_,
                                            orig_freq=sample_rate,
                                            new_freq=rate)
        torchaudio.save(outpath+name,waveform_,sample_rate=rate)
        
        print('-'*8+'success',name)
        
    else:
        print('-'*8+'ready',name)
        
if __name__ == '__main__':  
    
    rate = 22050
    outpath = 'audio_'+str(rate)+'/'
        
    if not os.path.isdir('audio_'+str(rate)):
        os.mkdir('audio_'+str(rate))
        
    names_ = [file.split('/')[1][:-4]+'.mp3' for file in sorted(glob('audio_rename/*'))]
    names = [c for c in names_ if not os.path.isfile(outpath+c[:-4]+'.mp3')]
    print('have {} data {} need proccess'.format(len(names_),len(names)))
    
    with Pool(20) as pool:  
        result = pool.map(re_sample,names)