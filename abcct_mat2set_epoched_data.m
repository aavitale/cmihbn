function abcct_mat2set_epoched_data_dev02(cfg, subj_id, cond_id)

% FUNCTION 
% import a 3d matrix (with dimension: channel, timepoint, trial)
% into a structure readable from eeglab (.set format)

% STILL TO ADD: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% - Line noise removal at 60 / 120 Hz (or low pass filter ??)
% - Baseline subtraction (before ICA) ??
% - average RE-REFERENCE ??
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


% INPUT:
%  subj_id = 'NDARAA898JB2_20180329'
%  cond_id = specific condition of interest:  i.e: 'f' for face processing

%  cfg: CONFIGURATION structure with this mandatory fields:
%     .project_dir         : where data, code and other folder are stored 
%     .eeglab_dir          : folder with all the eeglab functions (i.e: EEG\eeglab_20201226')
%     
%     .chanloc_struct      : structure with channel locations details
%                            (i.e.: EGI_129_chanloc_struct.mat)
%     .chan_toreject       : channel excluded from the initial dataset
%     .sample_rate         : in Hz (i.e: 1000); 
%     .n_sample            : number of timepoint (second dimension of the 3d matrix);
%     .time_min_sec        : epoch onset in second (i.e.: -0.2);

      % OPTIONAL - - - - - - - - - 
%     .continuous = 0;  % if 0 -> epoched data (3d matrix)
%     .chan_file           : .sfp file with channel location for 128 EGI layout
%

% OUTPUT:
%   the function will save one .set file for each field of the data with:
%   subj_ID +  field condition in the data structure 
%   example:  NDARAA898JB2_20180329_FaceInverted.set

% by andrea.vitale@gmail.com 20210527
%% 
% 

    %clear all
    %cfg = [];
    
    if isempty(cfg)
        cfg = [];
        %cfg.continuous = 0;  % if 0 -> epoched data (3d matrix)
        cfg.project_dir         = 'D:\IIT\_PROJECT\ABC_CT_EEG'
        %cfg.subj_id             = 'NDARAA898JB2_20180329'
        cfg.cond_id             = 'f'
        
        cfg.eeglab_dir          = 'D:\_TOOLBOX\eeglab_20201226'
        cfg.chanloc_struct      = 'EGI_129_chanloc_struct.mat'
        cfg.chan_toreject       = {'E125','E126'}
        
        cfg.sample_rate         = 1000; 
        cfg.n_sample            = 700;
        cfg.time_min_sec        = -0.2;
    end
    if isempty(subj_id)
        disp('subj_id is required !!!!')
        subj_id                 = 'NDARAA898JB2_20180329'
        cond_id                 = 'f'
    end        
    
    
%% ADD TOOLBOX

    % EEGLAB GUI:
    cd(cfg.eeglab_dir)
    eeglab
    %eeglab('nogui')
    
    do_chan_location = 1
    do_import_epoch = 1
    do_data_concat = 1
    do_run_ICA_concat = 1
    do_save = 0
    do_save_concat = 1
    
    
%% LOAD SUBJECT DATA
% and import in eeglab structure

    cd(fullfile(cfg.project_dir,'data'))
    data_struct = load([ subj_id '_' cond_id '.mat' ])
    
    
    struct_field = fieldnames(data_struct)
    
    for i_data = 1:length(struct_field)
        %data_tmp = data_struct.struct_field(i_data)
        eval([ 'data_tmp = data_struct.' struct_field{i_data} ';']);
        
        
        %% IMPORT 3d DATA INTO EEGLAB STRUCTURE - - - - - - - - - - -  - - - -
        eeg_struct = pop_importdata('dataformat','matlab','nbchan',0, 'data', data_tmp, ...
                             'setname', struct_field{i_data}, ...
                             'srate', cfg.sample_rate, 'pnts', cfg.n_sample, 'xmin', cfg.time_min_sec);
                             %'data','D:\\IIT\\_PROJECT\\ABC_CT_EEG\\data\\FaceInverted.mat',
        
        if do_chan_location
            
            load(fullfile(cfg.project_dir, 'data', cfg.chanloc_struct))
            n_chan = eeg_struct.nbchan;
            
            % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            % some more clever way of sub-selecting the channel has to be implemented
            % once we know the correct location of the channels
            chanloc_struct = chanloc_struct(1:n_chan);
            
            eeg_struct.chanlocs = chanloc_struct;        
            figure; topoplot([],eeg_struct.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', eeg_struct.chaninfo);
            
            eeg_struct = pop_select(eeg_struct, 'nochannel', cfg.chan_toreject); 
        end
        
        if do_import_epoch
            n_epoch = size(data_tmp,3);
            epoch_array = [];
            cond_cell = {};
            for i_epoch = 1:n_epoch
%                 epoch_array(i_epoch,1) = i_epoch;
%                 epoch_array(i_epoch,2) = i_data;
                
                %cond_array(i_epoch,1) = i_data;
                cond_cell{i_epoch,1} = struct_field{i_data};
            end
            
%             epoch_field = { 'epoch_array', 'cond_array'}
%             eeg_struct = pop_importepoch(eeg_struct, epoch_array, epoch_field)
            
            epoch_field = {'type'}
            eeg_struct = pop_importepoch(eeg_struct, cond_cell, epoch_field)
            %eeg_struct = pop_importepoch(eeg_struct, 'filename', epoch_array, 'fieldlist', epoch_field)
                        
%             % convert in txt format;
%             epoch_table = table(epoch_array, cond_array)
%             writetable(epoch_table,'epoch_tmp.txt');
%             %type epoch_tmp.txt
                                
        if do_save
            save_name = [ cfg.subj_id '_' struct_field{i_data} ]
            disp('saving...')
            pop_saveset(eeg_struct, 'filename', save_name)
        end
        
        % CONCATENATE - - - - - - - - - - - - - - 
        if do_data_concat
            if i_data > 1
                eeg_concat = pop_mergeset( eeg_concat, eeg_struct); %keepall
            else
                eeg_concat = eeg_struct;
            end
        end
        
        if do_save_concat
            save_name = [ subj_id '_' cond_id ]
            if i_data == length(struct_field)
                % copy epoch_type in the "type" column read by eeglab
                % (for subsequent epoching):
                %eeg_concat.event.type = eeg_concat.event.epochtype; 
                for i_epoch = 1:eeg_concat.trials
                    eeg_concat.event(i_epoch).type = eeg_concat.event(i_epoch).epochtype;
                end
                pop_saveset(eeg_concat, 'filename', save_name)
            end
        end                        
    end
    end  % FOR cycle across data segments
    
        
%% RUN (preprocessing step and) ICA on concatenated data
    if do_run_ICA_concat
        eeg_concat_ICA = pop_runica(eeg_concat, 'icatype', 'runica', 'extended',1,'interrupt','on');

        %%(PLOT component topography:)
        % (see: https://github.com/sccn/viewprops)
        %pop_topoplot(eeg_ICA, 0, [1:length(chan_toinclude)] ,'EDF file',[5 5] ,0,'electrodes','on');
        
        eeg_concat_ICA = pop_iclabel(eeg_concat_ICA, 'default');
        % for component viewing
        
        % REMOVE BAD COMPONENT based on ICLabels
        %eeg_nobadICA = pop_icflag(eeg_ICA, [NaN NaN;0.8 1;0.8 1;NaN NaN;0.8 1;0.8 1;0.8 1]);

        % if the % of brain ICA < 0.2 -> then is removed
        eeg_concat_nobadICA = pop_icflag(eeg_concat_ICA, [0 0.2;0.7 1;0.7 1;NaN NaN;0.7 1;0.7 1;0.7 1]);
        
        if do_save_concat
            save_name = [ subj_id '_' cond_id '_concat_ICA' ]
            %save_name = [ subj_id '_' cond_id '_concat_nobadICA' ]
            pop_saveset(eeg_concat_ICA, 'filename', save_name)
        end

        %% some PLOTs:
        close all
        figure;
        pop_spectopo(eeg_concat, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
        %pop_spectopo(eeg_cleanraw_avgref_nobadICA, 1, [ ], 'EEG' , 'percent', 50, 'freq', [8 13 20], 'freqrange',[2 80],'electrodes','on');
        save_name = [ subj_id '_' cond_id '_psd.jpg']
        saveas(gcf, save_name)
            
        pop_viewprops(eeg_concat_ICA, 0, [1:35], [2 80], [])
        save_name = [  subj_id '_' cond_id  '_concat_ICA.jpg' ]
        saveas(gcf, save_name)
        
    end
end %######################################################################

%%
%   SPLIT the DATA x conditions:
% EEG = pop_selectevent( EEG, 'type',{'FaceUpright'},'deleteevents','off','deleteepochs','on','invertepochs','off');

% EEG = pop_selectevent( EEG, 'type',{'FaceInverted'},'deleteevents','off','deleteepochs','on','invertepochs','off');

% create a STUDY:
% [STUDY ALLEEG] = std_editset( STUDY, [], 'name','face','commands',{{'index',1,'load','D:\\IIT\\_PROJECT\\ABC_CT_EEG\\data\\NDARAA898JB2_20180329_FaceUpright_ICA.set'},...
%                         {'index',2,'load','D:\\IIT\\_PROJECT\\ABC_CT_EEG\\data\\NDARAA898JB2_20180329_FaceInverted_ICA.set'},...
%                         {'index',1,'subject','s01'}, {'index',2,'subject','s01'}, ...
%                         {'index',1,'condition','face_up'}, {'index',2,'condition','face_inv'}},'updatedat','on','rmclust','on' );


% EEG = pop_rmbase( EEG,[-200 0] ,[]);

% [STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, {},'savetrials','on','interp','on','recompute','on','erp','on','erpim','on','erpimparams',{'nlines',10,'smoothing',10},'ersp','on','erspparams',{'cycles',[3 0.8] ,'nfreqs',100,'ntimesout',200},'itc','on');

% STUDY = pop_statparams(STUDY, 'condstats','on');
% STUDY = std_erpplot(STUDY,ALLEEG,'channels',{'E82','E83','E84','E85','E86','E87','E88','E89','E90','E91','E92','E93','E94','E95','E96','E97','E98'}, 'design', 1);
% STUDY = pop_erpparams(STUDY, 'plotconditions','together','averagechan','on');
