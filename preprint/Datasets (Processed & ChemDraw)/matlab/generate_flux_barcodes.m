function generate_flux_barcodes(csv_file_path)
% GENERATE_FLUX_BARCODES Creates publication-ready flux barcodes with smoothed overlay
%
% Usage:
%   generate_flux_barcodes()           % Opens file dialog to select CSV
%   generate_flux_barcodes(csv_file_path)  % Uses provided path
%
% Input:
%   csv_file_path - (Optional) Path to CSV file containing flux data
%                   Expected format: iteration,residue_index,residue_name,flux_value
%                   If not provided, opens file dialog
%
% Output:
%   Displays figure window - user can save manually using File menu or commands
%
% Features:
%   - Inverted grayscale colormap for flux intensity
%   - Smoothed line overlay from processed data
%   - Scale bar with units in kcal/mol·Å
%   - Publication-ready formatting

    % Handle input
    if nargin < 1 || isempty(csv_file_path)
        % Prompt user to select CSV file
        [file, path] = uigetfile('*.csv', 'Select flux data CSV file');
        if isequal(file, 0)
            error('No file selected. Operation cancelled.');
        end
        csv_file_path = fullfile(path, file);
        fprintf('Selected file: %s\n', csv_file_path);
    end
    
    if ~exist(csv_file_path, 'file')
        error('Input file does not exist: %s', csv_file_path);
    end
    
    % Load flux data
    fprintf('Loading flux data from: %s\n', csv_file_path);
    flux_data = readtable(csv_file_path);
    
    % Validate required columns
    required_cols = {'iteration', 'residue_index', 'flux_value'};
    missing_cols = setdiff(required_cols, flux_data.Properties.VariableNames);
    if ~isempty(missing_cols)
        error('Missing required columns: %s', strjoin(missing_cols, ', '));
    end
    
    % Extract data dimensions
    iterations = unique(flux_data.iteration);
    residues = unique(flux_data.residue_index);
    n_iterations = length(iterations);
    n_residues = length(residues);
    
    fprintf('Data dimensions: %d iterations × %d residues\n', n_iterations, n_residues);
    
    % Create flux matrix
    flux_matrix = zeros(n_iterations, n_residues);
    for i = 1:n_iterations
        for j = 1:n_residues
            mask = (flux_data.iteration == iterations(i)) & ...
                   (flux_data.residue_index == residues(j));
            if any(mask)
                flux_matrix(i, j) = flux_data.flux_value(mask);
            end
        end
    end
    
    % Get directory and construct processed data path
    [input_dir, ~, ~] = fileparts(csv_file_path);
    
    % Try to find processed data file for smoothed overlay
    % Look for pattern matching processed_flux_data files
    processed_file = '';
    possible_patterns = {
        '*processed_flux_data.csv',
        '*_processed_flux_data.csv',
        'processed_flux_data*.csv'
    };
    
    for pattern = possible_patterns
        files = dir(fullfile(input_dir, pattern{1}));
        if ~isempty(files)
            processed_file = fullfile(input_dir, files(1).name);
            break;
        end
    end
    
    % Create figure
    figure('Units', 'inches', 'Position', [1 1 12 8], 'Color', 'white');
    
    % Create barcode plot
    ax = axes('Position', [0.1 0.15 0.75 0.75]);
    
    % Plot flux barcode with inverted grayscale
    imagesc(residues, iterations, flux_matrix);
    
    % Set inverted grayscale colormap
    colormap(flipud(gray(256)));
    
    % Add colorbar with proper label
    cb = colorbar('Location', 'eastoutside');
    cb.Label.String = 'Flux (kcal/mol·Å)';
    cb.Label.FontSize = 12;
    cb.Label.FontWeight = 'bold';
    
    % Axis labels
    xlabel('Residue Number', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    title('Flux Intensity Barcode', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Set axis properties
    set(gca, 'YDir', 'normal');
    yticks(iterations);
    
    % Add grid
    grid on;
    set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.3);
    box on;
    
    % Add smoothed overlay if processed data exists
    hold on;
    
    % Display smoothed curve status
    fprintf('\n=== Smoothed Curve Display ===\n');
    
    if ~isempty(processed_file) && exist(processed_file, 'file')
        fprintf('Found processed data file: %s\n', processed_file);
        
        try
            % Read processed data
            proc_data = readtable(processed_file);
            
            % Handle multi-level headers by reading specific rows
            fid = fopen(processed_file, 'r');
            header1 = fgetl(fid);
            header2 = fgetl(fid);
            header3 = fgetl(fid);
            fclose(fid);
            
            % Try to extract residue IDs and flux values
            if contains(header1, 'signed_flux') || contains(header2, 'mean')
                % Read with proper options for multi-level headers
                opts = detectImportOptions(processed_file);
                opts.DataLines = [4 Inf];  % Start from row 4
                opts.VariableNamesLine = 3;
                proc_data = readtable(processed_file, opts);
                
                % Get residue IDs
                if ismember('residue_id', proc_data.Properties.VariableNames)
                    proc_residues = proc_data.residue_id;
                else
                    proc_residues = proc_data{:, 1};  % First column
                end
                
                % Get flux mean values (should be in column 2 based on structure)
                flux_mean = proc_data{:, 2};
                
                % Remove NaN values
                valid_idx = ~isnan(proc_residues) & ~isnan(flux_mean);
                proc_residues = proc_residues(valid_idx);
                flux_mean = flux_mean(valid_idx);
                
                % Apply KDE smoothing with sigma = 2
                sigma = 2;  % KDE bandwidth parameter
                
                % Create KDE smoothed curve
                % First, create a finer grid for smoother curve
                residue_fine = linspace(min(residues), max(residues), 500);
                
                % Initialize smoothed flux
                flux_smooth_fine = zeros(size(residue_fine));
                
                % Apply Gaussian kernel density estimation
                for i = 1:length(residue_fine)
                    % Calculate weighted sum using Gaussian kernel
                    weights = exp(-0.5 * ((proc_residues - residue_fine(i)) / sigma).^2);
                    weights = weights / sum(weights);
                    flux_smooth_fine(i) = sum(weights .* flux_mean);
                end
                
                % Interpolate back to original residue positions
                flux_smooth = interp1(residue_fine, flux_smooth_fine, residues, 'linear', 'extrap');
                
                % Create overlay axes
                ax2 = axes('Position', ax.Position);
                
                % Plot smoothed line
                plot(ax2, residues, flux_smooth, '-', 'Color', [1 0 0], ...
                    'LineWidth', 3, 'DisplayName', sprintf('KDE Smoothed (σ=%g)', sigma));
                
                % Configure overlay axes
                ax2.XLim = ax.XLim;
                ax2.XTick = [];
                ax2.YAxisLocation = 'right';
                ax2.Color = 'none';
                ax2.Box = 'off';
                ylabel(ax2, 'Mean Flux (kcal/mol·Å)', 'FontSize', 12, 'FontWeight', 'bold');
                
                % Scale Y-axis to match data range
                % Use automatic scaling to ensure curve is visible
                ylim(ax2, 'auto');
                
                % Make the curve more visible by adjusting the axis
                ax2.YLim = [min([flux_smooth(:); 0]), max(flux_smooth(:)) * 1.1];
                
                % Add legend
                legend(ax2, 'Location', 'northeast', 'FontSize', 10);
                
                fprintf('Successfully added smoothed overlay\n');
            else
                warning('Processed data file has unexpected format');
            end
            
        catch ME
            warning('Could not load processed data: %s', ME.message);
        end
    else
        fprintf('No processed data file found in directory: %s\n', input_dir);
        fprintf('Looking for files matching patterns:\n');
        for pattern = possible_patterns
            fprintf('  - %s\n', pattern{1});
        end
        fprintf('\nTo display smoothed curves, ensure processed flux data file is in the same directory.\n');
        fprintf('Expected file name should contain "processed_flux_data"\n');
    end
    
    fprintf('==============================\n\n');
    
    % Add scale bar
    add_scale_bar(ax, flux_matrix);
    
    % Removed residue markers for cleaner publication figure
    
    % Display save options
    fprintf('\nFigure generated successfully!\n');
    fprintf('To save the figure:\n');
    fprintf('  - Use File > Save As in the figure window\n');
    fprintf('  - Or use saveas(gcf, ''filename.png'')\n');
    fprintf('  - For high-resolution: print(gcf, ''filename.tiff'', ''-dtiff'', ''-r600'')\n');
    
end

function add_scale_bar(ax, flux_matrix)
% ADD_SCALE_BAR Adds a scale bar to the flux barcode plot
    
    % Calculate scale bar size (10% of flux range)
    flux_range = max(flux_matrix(:)) - min(flux_matrix(:));
    scale_length = round(flux_range * 0.1);
    
    % Position scale bar in bottom-left corner
    x_range = ax.XLim(2) - ax.XLim(1);
    y_range = ax.YLim(2) - ax.YLim(1);
    
    x_pos = ax.XLim(1) + 0.05 * x_range;
    y_pos = ax.YLim(1) + 0.05 * y_range;
    
    % Draw scale bar
    rectangle('Position', [x_pos, y_pos, x_range*0.02, scale_length], ...
        'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k', 'LineWidth', 1);
    
    % Add scale text
    text(x_pos + x_range*0.03, y_pos + scale_length/2, ...
        sprintf('%d kcal/mol·Å', scale_length), ...
        'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
    
end