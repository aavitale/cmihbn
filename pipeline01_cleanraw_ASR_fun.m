function pipeline01_cleanraw_ASR_fun(cfg, subj_name, file_set)

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

%   pipeline01_cleanraw_ASR_fun(cfg, 'NDARWC427JB2', file_set)

% = = = = = =  = = = = = = = = = 
%% MY CONFIGURATION structure / paths
if isempty(cfg)
    cfg.do_server = 0
    cfg.project_dir = 'C:\Users\Utente\Desktop\CMI_EEG_PREProcess'
    %cfg.project_dir = 'E:\CMI_EEG_PREProcess'
    cfg.data_set_dir = fullfile(cfg.project_dir, 'data_set')
    cfg.save_dir = fullfile(cfg.project_dir, 'data_pipeline01')
    
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
    % subj_name = 'NDARRZ199KNG'
    file_set = 'rs.set';
    % or - - - - - - - - - - -
    %file_set = 'desme.set';
end


% = = = = = = = = = = == = = = = = 
%% PARAMETERS
downsample_rate = 250 %Hz 

hpf_cutoff = 1 
lpf_cutoff = 80;
%lpf_cutoff = eeg_struct.srate/2 -1;

line_noise_freq = 60 %Hz 

do_chan_rejection = 1;
do_chan_interp_pruning = 1;
do_waveletICA = 0;    %<<< alternative a) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
do_cleanraw = 1;      %<<< alternative b) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
do_plot_chan = 0;
do_plot_PSD = 1;
    
%do_save_cleanline = 0
%do_save_wavclean_ICA = 1;
do_save_cleanraw_avgref_ICA = 1;

%do_save_wavclean_nobadICA = 1
do_save_cleanraw_avgref_nobadICA = 1;

do_save_fig = 1;
do_scorepoch = 0;
do_save_score = 1;


% = = =  = = = = = = = 
%% OPEN EEGLAB in NO GUI modality:
fprintf('... ADD TOOLBOX \n');

eeglab_dir = cfg.eeglab_dir   %fullfile(project_dir, 'tool', 'eeglab_20201226');
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


    % 26 CHANNELs to be REMOVED 
    chan_toreject = cfg.chan_toreject; 

    eeg_struct = pop_select(eeg_struct, 'nochannel', chan_toreject);
    eeg_raw_chanred = eeg_struct;


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
    
    % alternative b) - - - - - - - - - - - - - - - - - - - 
    %% cleanraw data + Artifact Subspace Reconstruction + avgref + ICA 
    % https://github.com/sccn/clean_rawdata/wiki
    % 
    %
    % ASR == good at removing occasional large-amplitude noise/artifacts
    % ICA == good at decomposing constant fixed-source noise/artifacts/signals
    %
    % vis_artifacts to compare the cleaned data to the original.
    % ----------------------------------------------------------------
    
%   chancorr_crit                       - Correlation threshold. If a channel is correlated at less than this value
%                                           to its robust estimate (based on other channels), it is considered abnormal in
%                                           the given time window. OPTIONAL, default = 0.8.
%   chan_max_broken_time                - Maximum time (either in seconds or as fraction of the recording) during which a 
%                                           retained channel may be broken. Reasonable range: 0.1 (very aggressive) to 0.6
%                                           (very lax). OPTIONAL, default = 0.5.
%   chan_detect_num_iter                - Number of iterations the bad channel detection should run (default = 10)
%   chan_detected_fraction_threshold	- Fraction how often a channel has to be detected to be rejected in the final
%                                           rejection (default 0.5)
%   flatline_crit                       - Maximum duration a channel can be flat in seconds (default 'off')


    if do_cleanraw
        close all;
        disp('... do CLEANRAW data \n')

        %  NOTCH FILTER (instead of CLEANLINE)
        eeg_notch = pop_eegfiltnew(eeg_struct, 'locutoff',line_noise_freq-2, ...
                                'hicutoff',line_noise_freq+2,'revfilt',1,'plotfreqz',1);
        %rethrow(eeg_notch)
        %pop_saveset(eeg_notch, 'filename', [ subj_name '_notch.set'])
        
        
        %----------------------------------------------------------------------
        % ITERATION of BAD SAMPLE and CHANNEL DETECTION
        %----------------------------------------------------------------------
        n_iter = 5;  %10
        bad_chan_table = zeros([n_iter, eeg_notch.nbchan]) -1;
        bad_sample_table = zeros([n_iter, eeg_notch.pnts]) -1;
        
        for i_iter = 1:n_iter
            %fprintf('... do CLEANRAW iteration on data already REPAIRED \n')
            %[eeg_cleanraw_iter, HP, BUR, bad_chan_cleanraw] = clean_artifacts(eeg_cleanraw_badchan_interp, ...
            % or 
            fprintf('... do CLEANRAW data on NOTCH filtered data \n')
            [eeg_cleanraw_iter, HP, BUR, bad_chan_cleanraw] = clean_artifacts(eeg_notch, ...
                'ChannelCriterion',0.8, ...
                'LineNoiseCriterion',4, ...
                'FlatlineCriterion',5, ...
                'BurstCriterion',20, ...
                'WindowCriterion','off', ...
                'Highpass','off', ...
                'BurstRejection','on');  % <<< only for the purpose of identifying bad samples <<<<<<
                %'BurstRejection','off');
        
            %vis_artifacts(eeg_cleanraw_iter,eeg_notch)
             
            % 0 means NO BAD channel/sample;  1 means BAD channel samples
            % (if some -1 value remains in the table.... something wrong !!
            bad_chan_table(i_iter,:) = ~eeg_cleanraw_iter.etc.clean_channel_mask;
            bad_sample_table(i_iter,:) = ~eeg_cleanraw_iter.etc.clean_sample_mask;
        end
        fprintf('!!! CLEANRAW iteration done .... \n')
            
        bad_chan_idx = find(sum(bad_chan_table,1) > 7);  % bad channel identified in more than 7 iteration
        bad_sample_idx = find(sum(bad_sample_table,1) > 7);  % bad channel identified in more than 7 iteration
        bad_sample_perc = sum(bad_sample_table,2) ./ eeg_notch.pnts *100
        
        % - - - - - - - - - - - - -  - -- -  -- - - - 
        % final ATTEMPT TO REPAIR the BAD SAMPLEs:
        [eeg_cleanraw, HP, BUR, bad_chan_cleanraw] = clean_artifacts(eeg_notch, ...
                'ChannelCriterion',0.8, ...
                'LineNoiseCriterion',4, ...
                'FlatlineCriterion',5, ...
                'BurstCriterion',20, ...
                'WindowCriterion','off', ...
                'Highpass','off', ...
                'BurstRejection','off');  % <<<<<< repair and not removal <<<<<<<<<<<<
            fprintf('!!! CLEANRAW data done .... \n')

        % use vis_artifact to look at the difference between clean and raw data
        %vis_artifacts(eeg_cleanraw,eeg_notch)
        %vis_artifacts(eeg_cleanraw_iter,eeg_cleanraw)
        %vis_artifacts(eeg_cleanraw_iter,eeg_cleanraw_badchan_interp)
        
        %bad_chan_cleanraw = eeg_cleanraw.etc.clean_channel_mask;
        bad_chan_label = {};
        xx = 1;
        for i_chan = 1:length(eeg_cleanraw.etc.clean_channel_mask)
            if ~eeg_cleanraw.etc.clean_channel_mask(i_chan)
                bad_chan_label{1,xx} = eeg_notch.chanlocs(i_chan).labels;
                xx = xx+1;
            end
        end
        
        % - - - - - - - - - - 
        % CHECK which portion of the data is still considered as artifactual (even after ASR)
        [eeg_cleanraw_ASR2, HP, BUR, bad_chan_cleanraw] = clean_artifacts(eeg_cleanraw, ...
                'ChannelCriterion',0.8, ...
                'LineNoiseCriterion',4, ...
                'FlatlineCriterion',5, ...
                'BurstCriterion',20, ...
                'WindowCriterion','off', ...
                'Highpass','off', ...
                'BurstRejection','on');  % 
            fprintf('!!! CLEANRAW + second run of ASR done .... \n')

        % some PLOTs = = =  = = = = = = = = = = 
         if do_save_fig
            
            %color_lim = [ 1 n_iter ];
            figure; set(gcf,'position',[10,10,1800,1000])
            subplot(231) 
            imagesc(bad_chan_table')
            xlabel('iteration'); ylabel('CHANNEL')
            %caxis(color_lim)
            title([' n° BAD CHAN = ' num2str(length(bad_chan_idx))])

            %subplot(234); topoplot( ~eeg_cleanraw_iter.etc.clean_channel_mask, eeg_cleanraw.chanlocs, ...
            subplot(234); topoplot( sum(bad_chan_table,1), eeg_notch.chanlocs, ...
                                    'electrodes','labelpoint','chaninfo',eeg_notch.chaninfo) ;
            %colormap(gca,'gray')
            title(bad_chan_label)

            subplot(2,3,2:3) 
            imagesc(bad_sample_table)
            ylabel('iteration'); xlabel('(BAD) SAMPLE  (10k samples = 40 sec)')
             title([' % of BAD SAMPLE = ' num2str(mean(bad_sample_perc))])

            % AFTER second round of ASR
            bad_sample_ASR2_table = double(~eeg_cleanraw_ASR2.etc.clean_sample_mask);
            subplot(3,3,8:9) 
    %         plot(sum(bad_sample_table,1)); 
    %         xlim([1 eeg_notch.pnts])
    %         ylabel({'n iteration';'where a sample is identified as BAD'})
            imagesc(bad_sample_ASR2_table); 
            colormap(gca,'gray')
            title('BAD segment  after initial ASR repair')
            
            cd(save_dir)
            save_name = [ subj_name '_' file_set(1:end-4) '_bad_sample&channel.jpg']
            saveas(gcf, save_name)
        end
            
        % = = = = = = = = = = = = = = = = = = = = = =
        % INTERPOLATE (before ICA ??) using also the channels to be pruned in the next step
        fprintf('... channel interpolation \n')
        eeg_cleanraw_badchan_interp = pop_interp(eeg_cleanraw, eeg_struct.chanlocs, 'spherical');
        % !!! also Cz is interpolated 
        
        % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        % disp('The function clean_rawdata has been deprecated and is only kept for backward compatibility');
        % disp('Use the clean_artifacts function instead.');
        
        % !!! for setting the appropriate parameters of the function see this paper:
        % https://pubmed.ncbi.nlm.nih.gov/31329105/
        % (see also: Comments to the HAPPE paper, and how to choose the critical parameters
            
        %         eeg_cleanraw = pop_clean_rawdata(eeg_notch, ...
        %                     'FlatlineCriterion',5, ...
        %                     'ChannelCriterion',0.8, ...
        %                     'LineNoiseCriterion',4, ...
        %                     'Highpass','off','WindowCriterion','off', ...
        %                     'BurstCriterion',20, 'BurstRejection','off','Distance','Euclidian');
        %                     % !!! burst criterion OFF -> in order to repair but not to remove (cut) datapoint
        %eeg_cleanraw = eeg_struct;
        
        %  RE-REFERENCE to the AVERAGE    
        eeg_cleanraw_avgref = pop_reref(eeg_cleanraw_badchan_interp, []);
        %eeg_avgref = eeg_struct;
    

        %% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        % further CHAN REDUCTION x ICA ??
        % pop_chanedit(eeg_cleanraw_avgref)
        % pop_eegbrowser(eeg_cleanraw_avgref)
        chan_interp_prune = cfg.chan_interp_prune;

        eeg_cleanraw_avgref = pop_select(eeg_cleanraw_avgref, 'nochannel', chan_interp_prune);
        % <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        %% ICA DECOMPOSITION (classification and rejection)
        eeg_cleanraw_avgref_ICA = pop_runica(eeg_cleanraw_avgref, 'icatype', 'runica', 'extended',1,'interrupt','on');

        %%(PLOT component topography:)
        % (see: https://github.com/sccn/viewprops)
        %pop_topoplot(eeg_ICA, 0, [1:length(chan_toinclude)] ,'EDF file',[5 5] ,0,'electrodes','on');
        
        % !!! to notice: according to KLUG, GRAMANN 2020:
        % 'default' classifier did not lead to good classification of muscular artifact, 'lite' was better overall.
        eeg_cleanraw_avgref_ICA = pop_iclabel(eeg_cleanraw_avgref_ICA, 'default');
                         % for component viewing
        if do_save_cleanraw_avgref_ICA
            %cd(fullfile(data_dir))
            cd(save_dir)
            
            save_name = [ subj_name '_' file_set(1:end-4) '_cleanraw_avgref_ICA.set']
  
            pop_saveset(eeg_cleanraw_avgref_ICA, 'filename', save_name)
            %pop_saveset(eeg_cleanraw_avgref_ICA, 'filename', [ subj_name '_' file_set '_cleanraw_avgref_ICA'])
        end
        
        % - - - -  - - - - - - - - - - - - - - - -
        % REMOVE BAD COMPONENT based on ICLabels
        %eeg_nobadICA = pop_icflag(eeg_ICA, [NaN NaN;0.8 1;0.8 1;NaN NaN;0.8 1;0.8 1;0.8 1]);

        % if the % of brain ICA < 0.2 -> then is removed
        eeg_cleanraw_avgref_nobadICA = pop_icflag(eeg_cleanraw_avgref_ICA, [0 0.2;0.7 1;0.7 1;NaN NaN;0.7 1;0.7 1;0.7 1]);

        n_badICA = length(find(eeg_cleanraw_avgref_nobadICA.etc.ic_classification.ICLabel.classifications(:,1) <= 0.2));
        disp('...clear raw + ASR + ICA done !!!')
        
        if do_save_cleanraw_avgref_nobadICA 
            cd(save_dir)
            save_name = [ subj_name '_' file_set(1:end-4) '_cleanraw_avgref_nobadICA.set']
            pop_saveset(eeg_cleanraw_avgref_nobadICA, 'filename', save_name);
        end
        
        
        % - - - -  - - - - - - - - - - - - - - - -
        % DIPFIT: https://eeglab.org/tutorials/09_source/DIPFIT.html
        % as in BEMOBIL pipeline
        
        
        % FIGURE = = = = = = = = = = =  == 
        if do_save_fig
            cd(save_dir)
            
            %figure; %subplot(211); pop_eegplot(eeg_notch, 1, 1, 1);
            %subplot(212); 
            pop_eegplot(eeg_cleanraw_avgref_nobadICA, 1, 1, 1);
            save_name = [ subj_name '_' file_set(1:end-4) '_cleanraw_avgref_nobadICA_scroll.jpg']
            saveas(gcf, save_name)

            close all
            pop_spectopo(eeg_notch, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
            %pop_spectopo(eeg_cleanraw_avgref_nobadICA, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
            save_name = [ subj_name '_' file_set(1:end-4) '_notch_psd.jpg']
            saveas(gcf, save_name)
            
            pop_viewprops(eeg_cleanraw_avgref_ICA, 0, [1:35], [2 80], [])
            %pop_viewprops(eeg_cleanraw_avgref_ICA, 0, [1:40], [2 80], [], 0, eeg_cleanraw_avgref_ICA.etc.ic_classification, 'on')
            %spec_opt, erp_opt, scroll_event, classifier_name, fig)
            
            save_name = [ subj_name '_' file_set(1:end-4) '_cleanraw_avgref_ICA.jpg']
            saveas(gcf, save_name)
        end
    end


catch ME
    disp(ME)
end


% if do_scorepoch
%     %% compute the SCORE for each prep step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % = ==========================================================
%     %% COMPUTE SCOREPOCHS at each preoprocessing step:
%     % = ==========================================================
%         %INPUT
%         %    cfg struct with the following fields
%         %           freqRange    - array with the frequency range used to compute the power
%         %                          spectrum (see MATLAB pwelch function)
%         %           fs           - integer representing sample frequency         
%         %           windowL      - integer representing the window length (in seconds)  
%         %           smoothFactor - smoothing factor for the power spectrum
%         cfg = []; 
%         % <<<<<<<<<<<<<<<<<< ENTIRE FREQUENCY RANGE<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%         cfg.freqRange = [ 2 : 80 ];
%         % <<<<<<<<<<<<<<<<<< THETA + ALPHA + BETA BAND <<<<<<<<<<<<<<<<<<<<<<<<<<<<
%         %cfg.freqRange = [ 2 : 40 ]; 
%         % <<<<<<<<<<<<<<<<<< only ALPHA BAND <<<<<<<<<<<<<<<<<<<<<<<<<<<<
%         %cfg.freqRange = [ 8 : 13 ]; 
% 
%         cfg.fs = eeg_struct.srate;
%         cfg.windowL = 2; % in sec <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%         cfg.smoothFactor = 0;
% 
%         prep_step = {
%                 'eeg_raw';
%                 'eeg_down';
%                 'eeg_hpf';
%                 'eeg_lpf';
% 
%     %             % alternative a) - - - - - - - - -
%     %     
%     %             'eeg_cleanline';
%     %             'eeg_psdthresh_badchan_interp';
%     %             'eeg_wavclean';
%     %             'eeg_wavclean_nobadICA';
%     %             'eeg_wavclean_nobadICA_avgref';
% 
%                 % alternative b) CLEANRAW + ASR + ICA - - - - - - - - -
% 
%                 'eeg_notch';
%                 'eeg_cleanraw';
%                 'eeg_cleanraw_badchan_interp';
%                 'eeg_cleanraw_avgref';
%                 'eeg_cleanraw_avgref_nobadICA';
% 
%                      };
% 
%         % CREATE A SCORE STRUCT for final report:
%         % with epoch not sorted !!!
%         score_struct = [];
% 
%         chan_toinclude = {};
%         for i_chan = 1:eeg_cleanraw_avgref_nobadICA.nbchan
%             %i_chan = 4
%             chan_toinclude{1,i_chan} = eeg_cleanraw_avgref_nobadICA.chanlocs(i_chan).labels;
%         end
% 
%         %or - - - - - - - -  -   
%         %chan_toinclude_18 = {'FP1' 'FP2' 'F3' 'F4' 'F7' 'F8' 'C3' 'C4' 'T3' 'T4' 'PZ' 'O1' 'O2' 'T5' 'T6' 'P3' 'P4' 'Fz' 
%         chan_toinclude_18 = {'E22' 'E9' 'E24' 'E124' 'E33' 'E122' 'E36' 'E104' 'E45' 'E108' 'E62' 'E58' 'E96' 'E52' 'E92' 'E70' 'E83' 'E11'};
%         chan_toinclude = chan_toinclude_18; 
% 
%         eeg_step = [];
%         for i_step = 1:length(prep_step)
%             eval(['eeg_step_tmp = ' prep_step{i_step} ]);
% 
%             % reduce to the minimum number of channels 
% 
%             if eeg_step_tmp.nbchan > length(chan_toinclude) %&& ...
%                     %~strcmp(prep_step{i_step}, 'eeg_cleanraw')
%                 try
%                     eeg_step_tmp =  pop_select(eeg_step_tmp, 'channel', chan_toinclude);
%                 catch ME
%                     disp(ME)
%                 end                
%             end
% 
% 
%             eval(['eeg_step.' prep_step{i_step} ' = eeg_step_tmp.data' ]);
% 
%             [idx_best_ep,epoch,score_Xep] = scorEpochs(cfg, eeg_step_tmp.data);
%             eval([ 'score_struct.' prep_step{i_step} '= score_Xep' ]);
%             score_epoch_mean(i_step,1) = mean(score_Xep);
%             disp(mean(score_Xep))
%             %bar(score_Xep)
%         end
% 
%         % REPORT other METRICS:::::::::::::::::::::::::::::::::::::::::::::::
%     %         n_chan; 
%         score_struct.n_chan_max          = n_chan_max;
%         score_struct.chan_toinclude      = chan_toinclude;
%         score_struct.bad_chan_cleanraw   = bad_chan_cleanraw; 
%         score_struct.n_badICA            = n_badICA; 
%         score_struct.score_epoch_mean    = score_epoch_mean;
%         score_struct.score_freqRange     = cfg.freqRange;
%         score_struct.score_windowL       = cfg.windowL;
% 
%     %         Percent_Variance_Kept_of_Post_Waveleted_Data=[];
% 
% 
%         if do_save_score
%             cd(save_dir)
%             save_name = [ subj_name '_' file_set(1:end-4) '_scorepoch_pipeline01.mat'];
%             if exist(save_name) > 0
%                 save_name = [ subj_name '_' file_set(1:end-4) '_chan18_scorepoch_pipeline01.mat']
%             end
%             save(save_name, 'score_struct')
% 
%             save_name = [ subj_name '_' file_set(1:end-4) '_eegstep_pipeline01.mat'];
%             save(save_name, 'eeg_step')
%         end
% end
% 
% disp('...end');
% % END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


