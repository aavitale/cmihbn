%function [eeg_step, score_struct] = pipeline02b_cleanline_wICA_fun(cfg, subj_name, file_set);
function pipeline02d_cleanline_wICA_fun(cfg, subj_name, file_set)

% by andrea.vitale@gmail.com 
% last update: 20210531


% EEGLAB version: 20201226
% PLUGIN required:
    % "EEGBrowser" v1.1
    % "ICLabel" v1.2.6
    % "PrepPipeline" v0.55.4
    % "clean_rawdata" v2.3 
    % "dipfit" v3.3 
    

% EXAMPLE of USAGE: 
%   cfg = []; 
%   subj_name = 'NDARWC427JB2'; 
%   file_set = 'rs.set';  % RESTING STATE DATA already converted in .set format  
%   pipeline02b_cleanline_wICA_fun(cfg, 'NDARWC427JB2', file_set)

% = = = = = =  = = = = = = = = = 
%% MY CONFIGURATION structure / paths
if isempty(cfg)
    cfg.do_server = 0
    
    cfg.project_dir = 'E:\CMI_EEG_PREProcess'
    cfg.data_set_dir = fullfile(cfg.project_dir, 'data_set')
    cfg.save_dir = fullfile(cfg.project_dir, 'data_pipeline02')
    
    cfg.eeglab_dir = fullfile(cfg.project_dir, 'tool', 'eeglab_20201226')
    
    
    cfg.chan_toreject = {
                'E127','E126',...
    'E25','E21','E17','E14','E8',...
    'E128','E32',          'E1','E125',...
    'E48','E43',            'E120','E119',...
    'E49',                          'E113',...
    'E56','E63',             'E99', 'E107',...
    'E68', 'E73', 'E81', 'E88', 'E94',...
               };
               
    %'E57' and 'E101' are considered the MASTOIDs and retained
    % Cz (online reference -> full of zeros  is retained)
    
    cfg.chan_interp_prune =  {
                          'E18','E10',...
                          'E38','E121', ...
                          'E44','E114', ...
                          'E28','E117',...
                          'E47','E98',...
                               }
    % channels that can be alternatively pruned:
        %'E35','E110',...
        %'E27','E123',...
        %'E39','E115'
        %'E68','E69','E73','E74','E81','E82','E88','E89','E94'}
    
end
% = = = = = =  = = = = = = = = = 

project_dir = cfg.project_dir;
data_set_dir = cfg.data_set_dir;
save_dir = cfg.save_dir;
if ~exist(save_dir); mkdir(save_dir); end
    
if isempty(file_set)
    file_set = 'rs.set';
    % or - - - - - - - - - - -
    %file_set = 'desme.set';
end


%% PARAMETERS
downsample_rate = 250 %Hz 

hpf_cutoff = 1 
lpf_cutoff = 80;
%lpf_cutoff = 45;  % <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%lpf_cutoff = eeg_struct.srate/2 -1;

line_noise_freq = 60 %Hz 

do_chan_rejection = 1;
do_chan_interp_pruning = 1; %<<<<if data are not concatenated <<<<<<<<<<<
do_waveletICA = 1;    %<<< alternative a) 
do_cleanraw = 0;      %<<< alternative b) 
    
do_plot_chan = 0;
do_plot_PSD = 1;
    
do_save_cleanline = 0
do_save_wavclean_ICA = 1;
%do_save_cleanraw_avgref_ICA = 1;

do_save_wavclean_nobadICA = 1
%do_save_cleanraw_avgref_nobadICA = 1;

do_save_fig = 1;
do_scorepoch = 0;
do_save_score = 1;


% = = =  = = = = = = = 
% %% OPEN EEGLAB in NO GUI modality:
fprintf('... ADD TOOLBOX \n');

eeglab_dir = cfg.eeglab_dir;
cd(eeglab_dir);
eeglab('nogui');

addpath(genpath(fullfile(project_dir, 'code')));


% = = = = = = = = = = = =
try 
    %% LOAD DATA set
    cd(data_set_dir)
    eeg_struct = pop_loadset('filename', [ subj_name '_' file_set ])
    eeg_raw = eeg_struct; 


    % SOME CHECKS - - - - - -
    sample_rate = eeg_struct.srate;
    % length in sec of the recording:
    n_sample = eeg_struct.pnts;
    n_sample / sample_rate;  %in sec
    n_chan = eeg_struct.nbchan;

    % number of channel that can be retained for ICA
    %(number of channel)^2 x 20 to 30 data points to perform ICA
    if n_sample > n_chan^2 * 20
        disp([ num2str ' channels can be given as input to ICA'])
    else
        n_chan_max = sqrt(n_sample/20);
        disp([ 'number of channels for ICA should be reduced to ' num2str(n_chan_max)])
    end
    

    % CHANNEL SELECTION - - - - - - - - - - - - - - 
    % subset of 11 channels as EOG
    chan_eye = {'E128', 'E32', 'E25', 'E21', 'E127', 'E17',...
            'E126', 'E14', 'E8', 'E1', 'E125'}

    eye_struct = pop_select(eeg_struct, 'channel', chan_eye);

    
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if do_chan_pruning
        %26 CHANNELs to be REMOVED 
        chan_toreject = cfg.chan_toreject;
        eeg_struct = pop_select(eeg_struct, 'nochannel', chan_toreject);

        % remove also ONLINE reference (Cz)  !!!! NO
        %eeg_struct = pop_select(eeg_struct, 'nochannel', {'Cz'});
        eeg_raw_chanrej = eeg_struct;

%     else % !!!!!!!!!! only 18 channels
%         chan_toinclude_18 = {'E22' 'E9' 'E24' 'E124' 'E33' 'E122' 'E36' 'E104' 'E45' 'E108' 'E62' 'E58' 'E96' 'E52' 'E92' 'E70' 'E83' 'E11'};
%         chan_toinclude = chan_toinclude_18; 
%         
%         eeg_struct = pop_select(eeg_struct, 'channel', chan_toinclude);
%         eeg_raw_chanpruned = eeg_struct;
    end


    %% - - - - - - - - - - - - - - - - - - - 
    % DOWNSAMPLE
    eeg_struct = pop_resample(eeg_struct, downsample_rate);
    eeg_down = eeg_struct;
    
    % BAND-PASS FILTERED data 
    fprintf('... BAND-PASS FILTERING \n')
    
    eeg_struct = pop_eegfiltnew(eeg_struct, hpf_cutoff, [], [],0,[],0);
    eeg_hpf = eeg_struct;
    
    eeg_struct = pop_eegfiltnew(eeg_struct, [], lpf_cutoff, [],0,[],0);
    eeg_lpf = eeg_struct;

    % vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    % -> same steps also for EOG channels ???

    % ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    % alternative a) - - - - - - - - - - - - - - - - - - - 
    %% cleanline + wICA + avgref (as in HAPPE)
    % !!! cleanline NOT working on the server
    % !!!(note: cleanline may not completely eliminate, 
    %           and re-referencing does not help)
    % for more uniformity we apply the same line noise removal than in the
    % alternative pipeline: notch filter
    % ----------------------------------------------------------------

    if do_waveletICA
        close all;
        disp('...do WAVELET ICA')
        
        %  NOTCH FILTER (instead of CLEANLINE)
        eeg_notch = pop_eegfiltnew(eeg_struct, 'locutoff',line_noise_freq-2, ...
                                'hicutoff',line_noise_freq+2,'revfilt',1,'plotfreqz',1);
        EEG = eeg_notch;
        
        
%         eeg_cleanline = pop_cleanline(eeg_struct, 'Bandwidth',2,'ChanCompIndices',[1:eeg_struct.nbchan] , ...
%           'SignalType','Channels','ComputeSpectralPower',true,'LineFrequencies',[line_noise_freq] , ...
%            'NormalizeSpectrum',false,'LineAlpha',0.01,'PaddingFactor',2,'PlotFigures',false,...
%           'ScanForLines',true,'SmoothingFactor',100,'VerbosityLevel',1,'SlidingWinLength',...
%            eeg_struct.pnts/eeg_struct.srate,'SlidingWinStep',eeg_struct.pnts/eeg_struct.srate);

%         eeg_cleanline = pop_cleanline(eeg_struct, 'bandwidth',2,'chanlist',[1:eeg_struct.nbchan] , ...
%             'computepower',1,'linefreqs', line_noise_freq, ...
%             'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype','Channels','tau',100,'verb',1,'winsize',4,'winstep',1);
%         EEG = eeg_cleanline;
        
            
        %% crude BAD CHANNEL DETECTION using spectrum criteria and 3SDeviations as channel outlier threshold, done twice
        EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[hpf_cutoff lpf_cutoff]); %1 125
        eeg_psdthresh_1 = EEG;
        
        EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[hpf_cutoff lpf_cutoff]);
        eeg_psdthresh_2 = EEG;
        
        bad_chan_psdthresh = eeg_cleanline.nbchan - eeg_psdthresh_2.nbchan; 
        
        % INTERPOLATE:
        EEG = pop_interp(EEG, eeg_struct.chanlocs, 'spherical');
        eeg_psdthresh_badchan_interp = EEG;
    
        
        %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        % further CHAN REDUCTION x ICA ??
        if do_chan_interp_pruning
            % pop_chanedit(eeg_cleanraw_avgref)
            % pop_eegbrowser(eeg_cleanraw_avgref)
            chan_interp_prune = cfg.chan_interp_prune;

            EEG = pop_select(EEG, 'nochannel', chan_interp_prune);
        end
        % <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       
        try
            disp('!!! start wICA ...')
            [wIC, A, W, IC] = wICA(EEG,'runica', 1, 1, EEG.srate, 5);
        catch wica_err
            if strcmp ('Output argument "wIC" (and maybe others) not assigned during call to "wICA".',wica_err.message)
            error('Error during wICA, most likely due to memory settings. Please confirm your EEGLAB memory settings are set according to the description in the HAPPE ReadMe')
            else
            rethrow(wica_err)
            end
        end
        disp('wICA done!!!')

        %reconstruct artifact signal as channelsxsamples format from the wavelet coefficients
        artifacts = A*wIC;
    
        %reshape EEG signal from EEGlab format to channelsxsamples format
        EEG2D=reshape(EEG.data, size(EEG.data,1), []);
    
        %subtract out wavelet artifact signal from EEG signal
        EEG.data = EEG2D-artifacts;
        %eeg_wavclean = EEG2D-artifacts;

        eeg_wavclean = EEG; 

        
        % RE-REFERENCE TO THE AVERAGE - - - -  - - - - - - - - - - -  -
        %(in the original HAPPE pipeline reref is after ICA)
        eeg_wavclean_avgref = pop_reref(eeg_wavclean, []);
        
        
        % ICA DECOMPOSITION (classification + rejection)
        eeg_wavclean_avgref_ICA = pop_runica(eeg_wavclean_avgref, 'icatype', 'runica', 'extended',1,'interrupt','on');
        
        eeg_wavclean_avgref_ICA = pop_iclabel(eeg_wavclean_avgref_ICA, 'default');
        %pop_viewprops(eeg_wavclean_ICA, 0, [1:eeg_wavclean_ICA.nbchan], [2 80]) % for component properties
    
        if do_save_wavclean_ICA
            %cd(fullfile(project_dir, 'data_prep'))
            cd(save_dir)
            pop_saveset(eeg_wavclean_avgref_ICA, 'filename', [ subj_name '_' file_set(1:end-4) '_wavclean_avgref_ICA'])
        end
                        
        eeg_wavclean_avgref_nobadICA = pop_icflag(eeg_wavclean_avgref_ICA, [0 0.2;0.7 1;0.7 1;NaN NaN;0.7 1;0.7 1;0.7 1]);
        n_badICA = length(find(eeg_wavclean_avgref_nobadICA.etc.ic_classification.ICLabel.classifications(:,1) <= 0.2));
       
        % !!! NO 
        %eeg_wavclean_nobadICA_avgref = pop_reref(eeg_wavclean_nobadICA, []);
        
        if do_save_wavclean_nobadICA 
            %pop_saveset(eeg_wavclean_nobadICA_avgref, 'filename', [ subj_name '_' file_set(1:end-4) '_wavclean_nobadICA_avgref'])        
            pop_saveset(eeg_wavclean_avgref_nobadICA, 'filename', [ subj_name '_' file_set(1:end-4) '_wavclean_avgref_nobadICA'])        
        end
    end
       
    % FIGURE = = = = = = = = = = =  ==
    if do_save_fig
         cd(save_dir)
         
        %pop_eegplot(eeg_wavclean_nobadICA_avgref, 1, 1, 1);
        %save_name = [ subj_name '_' file_set(1:end-4) '_wavclean_nobadICA_avgref_scroll.jpg']
        
        % wavelet ICA: IC - wIC
        save_name = [ subj_name '_' file_set(1:end-4) '_wavclean_scroll.jpg']
        saveas(gcf, save_name)
        
        close all
        pop_spectopo(eeg_cleanline, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
        %pop_spectopo(eeg_cleanraw_avgref_nobadICA, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
        save_name = [ subj_name '_' file_set(1:end-4) '_cleanline_psd.jpg']
        saveas(gcf, save_name)
        
        if eeg_wavclean_avgref_ICA.nbchan > 35; n_ICA = 35;
        else n_ICA = eeg_wavclean_avgref_ICA.nbchan;
        end
        pop_viewprops(eeg_wavclean_avgref_ICA, 0, [1:n_ICA], [2 80], [])
        %pop_viewprops(eeg_cleanraw_avgref_ICA, 0, [1:40], [2 80], [], 0, eeg_cleanraw_avgref_ICA.etc.ic_classification, 'on')
        %spec_opt, erp_opt, scroll_event, classifier_name, fig)
       
        save_name = [ subj_name '_' file_set(1:end-4) '_wavclean_avgref_ICA.jpg']
        saveas(gcf, save_name)
    end


catch ME
    disp(ME)
end


% %% compute the SCORE for each prep step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % = ==========================================================
% %% COMPUTE SCOREPOCHS at each preoprocessing step:
% % = ==========================================================
%     %INPUT
%     %    cfg struct with the following fields
%     %           freqRange    - array with the frequency range used to compute the power
%     %                          spectrum (see MATLAB pwelch function)
%     %           fs           - integer representing sample frequency         
%     %           windowL      - integer representing the window length (in seconds)  
%     %           smoothFactor - smoothing factor for the power spectrum
%     cfg = []; 
%     % <<<<<<<<<<<<<<<<<< ENTIRE FREQUENCY RANGE<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%     cfg.freqRange = [ 2 : lpf_cutoff ];
%     % <<<<<<<<<<<<<<<<<< THETA + ALPHA + BETA BAND <<<<<<<<<<<<<<<<<<<<<<<<<<<<
%     %cfg.freqRange = [ 2 : 40 ]; 
%     % <<<<<<<<<<<<<<<<<< only ALPHA BAND <<<<<<<<<<<<<<<<<<<<<<<<<<<<
%     %cfg.freqRange = [ 8 : 13 ]; 
%   
%     cfg.fs = eeg_struct.srate;
%     cfg.windowL = 2; % in sec <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%     cfg.smoothFactor = 0;
% 
%     prep_step = {
%             %'eye_struct'; 
%             'eeg_raw_chanpruned';
%             'eeg_hpf';
%             'eeg_lpf';
%             
%             % alternative a) CLEANLINE + wICA + avgref (HAPPE) - - - - - - - - -
%     
%             'eeg_cleanline';
%             'eeg_psdthresh_1';
%             'eeg_psdthresh_2';
%             'eeg_psdthresh_badchan_interp';
%             'eeg_wavclean';
%             'eeg_wavclean_avgref';
%             'eeg_wavclean_avgref_nobadICA';
%             
% %             % alternative b) CLEANRAW + ASR + ICA - - - - - - - - -
% %     
% %             'eeg_notch';
% %             'eeg_cleanraw';
% %             'eeg_cleanraw_badchan_interp';
% %             'eeg_cleanraw_avgref';
% %             'eeg_cleanraw_avgref_nobadICA';
%                  };
%              
%     % CREATE A SCORE STRUCT for final report:
%     % with epoch not sorted !!!
%     score_struct = [];
%     
%     if do_chan_pruning
%         chan_toinclude = {};
%         for i_chan = 1:eeg_wavclean_avgref_nobadICA.nbchan
%             %i_chan = 4
%             chan_toinclude{1,i_chan} = eeg_wavclean_avgref_nobadICA.chanlocs(i_chan).labels;
%         end
%     end
%     
% %     %or - - - - - - - -  -   
% %     %chan_toinclude_18 = {'FP1' 'FP2' 'F3' 'F4' 'F7' 'F8' 'C3' 'C4' 'T3' 'T4' 'PZ' 'O1' 'O2' 'T5' 'T6' 'P3' 'P4' 'Fz' 
% %     chan_toinclude_18 = {'E22' 'E9' 'E24' 'E124' 'E33' 'E122' 'E36' 'E104' 'E45' 'E108' 'E62' 'E58' 'E96' 'E52' 'E92' 'E70' 'E83' 'E11'};
% %     chan_toinclude = chan_toinclude_18; 
% 
%     
%     eeg_step = [];
%     for i_step = 1:length(prep_step)
%         eval(['eeg_step_tmp = ' prep_step{i_step} ]);
%         
%         % reduce to the minimum number of channels 
%         if eeg_step_tmp.nbchan > length(chan_toinclude) %&& ...
%             %~strcmp(prep_step{i_step}, 'eeg_cleanraw')
%             try
%                 eeg_step_tmp =  pop_select(eeg_step_tmp, 'channel', chan_toinclude);
%             catch ME
%                 disp(ME)
%             end
%         end
%         
%         eval(['eeg_step.' prep_step{i_step} ' = eeg_step_tmp.data' ]);
%         
%         [idx_best_ep,epoch,score_Xep] = scorEpochs(cfg, eeg_step_tmp.data);
%         eval([ 'score_struct.' prep_step{i_step} '= score_Xep' ]);
%         score_epoch_mean(i_step,1) = mean(score_Xep);
%         disp(mean(score_Xep))
%         %bar(score_Xep)
%     end
%   
%     % REPORT other METRICS:::::::::::::::::::::::::::::::::::::::::::::::
% %         n_chan; 
%     score_struct.n_chan_max          = n_chan_max;
%     score_struct.chan_toinclude      = chan_toinclude;
%     score_struct.bad_chan_psdthresh  = bad_chan_psdthresh; %<<<<<<<<<<<<<<<
%     score_struct.n_badICA            = n_badICA; 
%     score_struct.score_epoch_mean    = score_epoch_mean;
%     score_struct.score_freqRange     = cfg.freqRange;
%     score_struct.score_windowL       = cfg.windowL;
%     
% %         Percent_Variance_Kept_of_Post_Waveleted_Data=[];
% 
%      if do_save_score
%         cd(save_dir)
%         save_name = [ subj_name '_' file_set(1:end-4) '_scorepoch_pipeline02.mat'];
% %         if exist(save_name) > 0
% %             save_name = [ subj_name '_' file_set(1:end-4) '_chan18_scorepoch_pipeline01.mat']
% %         end
%         save(save_name, 'score_struct')
%         
%         save_name = [ subj_name '_' file_set(1:end-4) '_eegstep_pipeline02.mat'];
%         save(save_name, 'eeg_step')
%      end
%     
% disp('...end');
% % END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


