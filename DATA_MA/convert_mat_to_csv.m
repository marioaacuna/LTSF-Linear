% filepath: /home/mario/Documents/Transformer_AR_test/convert_mat_to_csv.m
function convert_mat_to_csv(mat_filename, output_filename)
    % Load the .mat file and get the first variable regardless of its name
    data = load(mat_filename);
    varNames = fieldnames(data);
    t_b = data.(varNames{1});  % using the first variable in the file
    
    % Calculate time vector assuming 12.5 Hz frame rate
    [n, m] = size(t_b);
    secondsVec = (0:n-1)' / 12.5;
    
    % Convert seconds to datetime format using base date 2016-07-01 00:00:00
    baseDate = datetime('2016-07-01 00:00:00','InputFormat','yyyy-MM-dd HH:mm:ss');
    Date = baseDate + seconds(secondsVec);
    Date.Format = 'yyyy-MM-dd HH:mm:ss.SSS';
    
    % Create table with Date as first column and features as F1, F2, ... Fn
    T = array2table(t_b, 'VariableNames', strcat('F', string(1:m)));
    T = addvars(T, Date, 'Before', 1, 'NewVariableNames', 'date');
    
    % Save the table as CSV file
    writetable(T, output_filename);
end