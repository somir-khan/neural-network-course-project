%% Generate DWT Coefficients

files = dir("../dataset/raw/");
% lpFilters = importdata("models/lp.txt");
decimationRate=1;
num_levels = 6;
%decimation rate 1 means no downsampling and 2 means 50Hz downsampling
%mkdir("../dataset/DwtProcessed"+num2str(100/1)+"/Dwt_Co_no_down/");
mkdir("../dataset/Dwtprocessed"+num2str(100/decimationRate)+"/"+num_levels+"-level/");
% mkdir("./dataset/processed"+num2str(100/decimationRate)+"/features/");
for i = {files.name}
    i = string(i);
    disp(i)
    if strlength(i) == 13 
        mat = importdata("../dataset/raw/"+i);
        tic;
        downsampled = {};
        for j = mat
            j = filter(ones(1,decimationRate)/decimationRate,1,j);
%                 j = j/std(j);               
            downsampled =  cat(1,downsampled,j(1:decimationRate:end,1));            
        end
        disp(class(downsampled))
        mat = horzcat(downsampled{:});
        %disp(class(mat))
        fprintf('The shape of A is %d x %d\n', size(mat));
        %error('cccc')
        Dwt = {};
        for k = 1:18
            disp(size(mat, 2))
            [c,l] = wavedec(mat(:, k), num_levels, 'db4');
            Dwt{k} = wrcoef('d',c,l,'db4',num_levels);
            %subplot(5,1,1); plot(mat(:, k)); title('Original Signal');
            %subplot(5,1,2); plot(d3); title('Approximation Coefficients (Level 3)');
            %error('yo')
        end
        Dwt = horzcat(Dwt{:});
        %disp(mat-Dwt)
        %fprintf('The shape of A is %d x %d\n', size(mat-Dwt));
        %error('yo')
        csvwrite("../dataset/Dwtprocessed"+num2str(100/decimationRate)+"/"+num_levels+"-level/"+i,Dwt);
        disp('done')
        %csvwrite("../dataset/processed"+num2str(100/decimationRate)+"/AR_force/"+i,mat);
    end
end