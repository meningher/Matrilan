function [] = correlation_matrices(sum_ID,PP_folder,output_folder,naming_convention)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% defining input and output folders and how many numbers are in the ID
if nargin<3
    
    prompt = {'Enter the length of the ID number:', 'Enter intensity data directory path:', 'Enter output directory path:','Enter naming convention (e.g. PB):','Enter window size (in sec):','Enter lag (# of vols):','Enter TR:'};
    dlgtitle = 'Input';
    dims = [1 150];
    definput = {'6',...
        '/media/neuro/LivnyLab/Research/TBI_magneton/Analyses/MatriLan/matrices/Mat_180521/FC/Intensity',...
        '/media/neuro/LivnyLab/Research/TBI_magneton/Analyses/MatriLan/matrices/Mat_180521/FC',...
        'PA',...     % 1= Brainnetome, 2=AAL, 3= Tomer's atlas
        '60',...
        '1',...
        '2'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    
    sum_ID = str2double(answer{1});
    PP_folder = answer{2};
    output_folder = answer{3};
    naming_convention = answer{4};
    window_size = str2double(answer{5});
    lag = str2double(answer{6});
    TR = str2double(answer{7});
    
    
    
end



%% creating a new folder for the output

Int_folder = PP_folder; % writing the path of the input folder
mkdir (output_folder, 'Dynamic_Correlations'); % creates an empty directory
output_folder = [output_folder '/Dynamic_Correlations']; % defining the output path


%% choosing the r p z values
list = {'Pearson Correlation (r)','p values','Fisher Transformation (Z)'};

[indx_statistic,selection_logical] = listdlg('PromptString',{'Choose your output values.',...
    'you can choose single or multiple values.',''},'ListString',list, 'ListSize',[300,80]);


%% choosing the THRESHOLD
list = {'No Threshold','Absolute Threshold (which you define)','0.05 Threshold','FDR Threshold'};

[indx_threshold,selection_logical] = listdlg('PromptString',{'Choose your threshold:'},'ListString',list, 'ListSize',[300,80]);


%% If the 'abs_TH' was chosen then the user is asked to choose a TH

absolute_threshold=0.2; % default value

if ismember(2, indx_threshold)==1
    
    prompt = {'Enter the wanted absolute threshold:'};
    dlgtitle = 'Input';
    dims = [1 30];
    definput = {'0.2'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    
    absolute_threshold = str2double(answer{1});
end


%% choosing the matrix order
list = {'AAL- Hemispheric','AAL- original order (LRLR)','AAL- Hemispheric- no cerebellum','Brainnetome- Hemispheric', 'Enter your own order (vector)'};

[indx_order,selection_logical] = listdlg('PromptString',{'Choose your matrix order- choose only one :'},'ListString',list, 'ListSize',[300,80]);


%% If the user chose to enter his own order vector- he enters it here

if ismember(5, indx_order)==1
    
    prompt = {'Enter your vector here in square parantheses (e.g. [0.2, 0.3], it must be in the same length as your atlas.txt file) '};
    dlgtitle = 'Input';
    dims = [1 70];
    definput = {'0.2'};
    answer = inputdlg(prompt,dlgtitle,dims,definput);
    
    user_order = eval(answer{1});
end


%% deleting the diagonal choice
list = {'Yes','No - keep the diagonal'};

[indx_diagonal,selection_logical] = listdlg('PromptString',{'would you like to delete the diagonal?'},'ListString',list, 'ListSize',[300,80]);%% creating the matrices

%% create folders for each type of matrix

value_names={'r_val','p_val','z_val'};
th_names={'no_TH','TH_absolute','TH_0.05','TH_FDR'};


for ruth = 1: length(indx_statistic)
    if indx_statistic(ruth)==1
        
        mkdir (output_folder, value_names{1}); % creates an empty directory
        outputpath{ruth}=[output_folder '/' value_names{1}];
        
    elseif indx_statistic(ruth)==2
        
        mkdir (output_folder, value_names{2}); % creates an empty directory
        outputpath{ruth}=[output_folder '/' value_names{2}];
        
    elseif indx_statistic(ruth)==3
        
        mkdir (output_folder, value_names{3}); % creates an empty directory
        outputpath{ruth}=[output_folder '/' value_names{3}];
        
    end
end

for ferguson = 1: length(indx_threshold)
    if indx_threshold(ferguson)==1
        
        for aly = 1:length(indx_statistic)
            mkdir (outputpath{aly}, th_names{1}); % creates an empty directory
        end
        
    elseif indx_threshold(ferguson)==2
        
        for aly = 1:length(indx_statistic)
            mkdir (outputpath{aly}, th_names{2}); % creates an empty directory
        end
        
    elseif indx_threshold(ferguson)==3
        
        for aly = 1:length(indx_statistic)
            mkdir (outputpath{aly}, th_names{3}); % creates an empty directory
        end
        
    elseif indx_threshold(ferguson)==4
        
        for aly = 1:length(indx_statistic)
            mkdir (outputpath{aly}, th_names{4}); % creates an empty directory
        end
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CREATING THE MATRIX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder_info=dir(Int_folder);  % turns the folder of all the subjects to a struct
window_size_vols= fix(window_size/TR); % the number of volumes in each sliding window


for jess = 3:3(folder_info) % runs on the subjects rest scans
    if contains(folder_info(jess).name,'Int') &&  contains(folder_info(jess).name,'.csv')% looking for intensity csv files
        
        int_path=[Int_folder '/' folder_info(jess).name];
        intensity_mat=importdata(int_path); % loading a subjects intensity matrix
        num_of_vols = size(intensity_mat,2); % number of vols in the entire scan
        num_of_wind= num_of_vols/lag-(window_size_vols-lag); % calculating the number of wanted windows for each sub
        
        for window = 1:num_of_wind %running on windows
            
            mini_int_mat=intensity_mat(:,(window-1)*lag+1:(window-1)*lag+window_size_vols);
            disp(mini_int_mat)
            % r and p unordered
            [r_mat, p_mat]= corrcoef(mini_int_mat'); %%%%%%%%%%%%%CHANGEDDDDDDDDD FROM CORR TO CORRCOEF 8.10.20
            
            
            %% !!!!!!!!! THRESHOLDING !!!!!!!!!!!!
            
            % ABS THRESHOLD
            r_abs=r_mat;
            p_abs=p_mat;
            r_abs(r_mat<=absolute_threshold)=0;% creating the TH
            p_abs(r_mat<=absolute_threshold)=0;
            
            % 0.05 THRESHOLD
            r_005=r_mat;
            p_005=p_mat;
            r_005(p_mat>=0.05)=0; % creating the TH
            p_005(p_mat>=0.05)=0;
            
            % FDR THRESHOLD
            if ismember(4, indx_threshold)==1
                r_FDR=r_mat;
                p_FDR=p_mat;
                
                pvals=triu(p_mat,1); %deleting the diagonal and the lower triangle
                pvals=nonzeros(pvals); % deleting all the zeroes and creating a vector
                sorted_pvals=sort(pvals); % sorting the pvals in a descending order
                
                index_vec=(1:length(sorted_pvals)); % creating an ascending index vector
                correction_vec=index_vec.*0.05./length(sorted_pvals); % creating an ascending correction vector
                
                FDR_TH=0.05;
                for cece=length(sorted_pvals):-1:1 % running from the biggest value to the smallest- finding the biggest P val that is smaller than its TH
                    
                    if sorted_pvals(cece)<correction_vec(cece) % looking for the first significant p val
                        
                        FDR_TH=sorted_pvals(cece); % redefining TH
                        break
                        
                    end
                    
                end
                
                disp(['The TH for sub' folder_info(jess).name(1:sum_ID+3) ' is: ' num2str(FDR_TH)]); % sanity check
                
                % FDR thresholded unordered
                r_FDR(p_mat>=FDR_TH)=0; % creating the TH
                p_FDR(p_mat>=FDR_TH)=0; % creating the TH
            end
            
            
            %% mat order vectors
            
            order.hemispheric{1}= [1:2:108 2:2:108 109:116]'; % AAL atlas 116- the hemispheric order: L, then R, then vermis
            order.hemispheric{2}= [1:116]'; % AAL atlas 116- LRLR
            order.hemispheric{3}= [1:2:90 2:2:90]'; % AAL atlas 90- L, then R, no cerebellum
            order.hemispheric{4}= [1:2:250 251:3:274 2:2:250 253:3:274 252:3:274]'; % brainnetome atlas 274 L, then R, then vermis
            
            
            %% choosing the hemispheric order
            
            if ismember(5, indx_order)
                hemispheric_order= user_order;
            else
                hemispheric_order = order.hemispheric{indx_order};
            end
            
            %% saving all matrices in a struct
            
            mat.r_val(1).mat=r_mat;
            mat.r_val(1).name='correlation';
            
            mat.r_val(2).mat=r_abs;
            mat.r_val(2).name=['correlation abs TH ' num2str(absolute_threshold)];
            
            mat.r_val(3).mat=r_005;
            mat.r_val(3).name='correlation 0.05 TH';
            
            
            mat.p_val(1).mat=p_mat;
            mat.p_val(1).name='p values';
            
            mat.p_val(2).mat=p_abs;
            mat.p_val(2).name=['p values abs TH ' num2str(absolute_threshold)];
            
            mat.p_val(3).mat=p_005;
            mat.p_val(3).name='p values 0.05 TH';
            
            
            mat.z_val(1).mat=atanh(r_mat);
            mat.z_val(1).name='z correlations';
            
            mat.z_val(2).mat=atanh(r_abs);
            mat.z_val(2).name=[ 'z correlations abs TH ' num2str(absolute_threshold)];
            
            mat(1).z_val(3).mat=atanh(r_005);
            mat(1).z_val(3).name='z correlations 0.05 TH';
            
            
            
            if ismember(4, indx_threshold)==1
                
                mat.r_val(4).mat=r_FDR;
                mat.r_val(4).name=['correlations FDR TH: ' num2str(FDR_TH)];
                
                mat.p_val(4).mat=p_FDR;
                mat.p_val(4).name=[ 'p values FDR TH: ' num2str(FDR_TH)];
                
                mat.z_val(4).mat=atanh(r_FDR);
                mat.z_val(4).name=['z correlations FDR TH: ' num2str(FDR_TH)];
                
            end
            
            
            
            value_names={'r_val','p_val','z_val'};
            th_names={'no_TH','TH_absolute','TH_0.05','TH_FDR'};
            
            
            
            %% ordering all hemispheres + deleting diagonal if needed
            %%  saving the matrices
            for winston = indx_statistic
                for coach = indx_threshold
                    
                    matrix= eval(['mat.' value_names{winston} '(' num2str(coach) ').mat']); % opens the appropriate matrix
                    ord_mat=matrix(hemispheric_order,hemispheric_order); % reorganizing the matrix
                    
                    %% delete diagonal
                    if ismember(1, indx_diagonal)==1
                        
                        ord_mat(~~eye(length(matrix)))=0; %deleting the diagonal
                    end
                    
                    individual_output_path=[output_folder '/' value_names{winston} '/' th_names{coach}];
                    folder_name=[ individual_output_path '/' folder_info(jess).name(1:sum_ID+3) '_' num2str(window_size_vols) '_' num2str(lag) '_Tapered' ];
                    
                    if ~exist(folder_name, 'dir')
                    mkdir(folder_name); % creates an empty directory of the subject's name
                    end
                    
                    figure ('Visible','off')
                    %subplot(2,1,1)
                    imagesc(ord_mat), colorbar; axis image;
                    name=eval(['mat.' value_names{winston} '(' num2str(coach) ').name']);
                    xlabel(name)% the connectome pearson correlations unthresholded
                    
                    print([ folder_name '/' folder_info(jess).name(1:sum_ID+3) '_' name '_' naming_convention '_' num2str(window_size_vols) '_' num2str(lag) '_' num2str(window)],'-djpeg','-r1000');
                    
                    csvwrite([ folder_name '/' folder_info(jess).name(1:sum_ID+3) '_' name '_' naming_convention '_' num2str(window_size_vols) '_' num2str(lag) '_' num2str(window) '.csv'],ord_mat); % outputting the corr thresholded ordered by hemisphere
                    close all
                end
            end
        end
    end
end

end
