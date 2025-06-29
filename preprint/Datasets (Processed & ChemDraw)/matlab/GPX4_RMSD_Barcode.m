% MATLAB script for RMSD Barcode between GPX4 WT (6HN3) and GPX4–ML162 (6HKQ)
pdb_wt = getpdb('7L8K');    % originally 6hn3
pdb_ml162 = getpdb('7U4K'); % originally 6hkq

% Extract CA atoms only (single entry per residue)
ca_wt_all = pdb_wt.Model.Atom(strcmp({pdb_wt.Model.Atom.AtomName},'CA'));
ca_ml162_all = pdb_ml162.Model.Atom(strcmp({pdb_ml162.Model.Atom.AtomName},'CA'));

% Create unique residue list
[wt_residues, ia_wt] = unique([ca_wt_all.resSeq],'stable');
[ml162_residues, ia_ml162] = unique([ca_ml162_all.resSeq],'stable');

% Extract unique CA atoms
ca_wt = ca_wt_all(ia_wt);
ca_ml162 = ca_ml162_all(ia_ml162);

% Identify common residues
common_residues = intersect(wt_residues, ml162_residues);

% Preallocate coordinates
coords_wt = zeros(length(common_residues), 3);
coords_ml162 = zeros(length(common_residues), 3);

% Assign coordinates safely
for i = 1:length(common_residues)
    idx_wt = find([ca_wt.resSeq] == common_residues(i), 1, 'first');
    idx_ml162 = find([ca_ml162.resSeq] == common_residues(i), 1, 'first');
    
    coords_wt(i,:) = [ca_wt(idx_wt).X, ca_wt(idx_wt).Y, ca_wt(idx_wt).Z];
    coords_ml162(i,:) = [ca_ml162(idx_ml162).X, ca_ml162(idx_ml162).Y, ca_ml162(idx_ml162).Z];
end

% Align structures with Procrustes analysis
[~, Z, ~] = procrustes(coords_wt, coords_ml162, 'Scaling', false, 'Reflection', false);

% RMSD per residue
diff_coords = coords_wt - Z;
residue_rmsd = sqrt(sum(diff_coords.^2, 2));

% Barcode plot (inverse grayscale)
figure('Position', [100, 100, 1200, 150]);
imagesc(residue_rmsd');
colormap(flipud(gray)); % Inverted grayscale
colorbar;
caxis([0, 2]);
xlabel('Residue Number');
set(gca, 'YTick', []);
set(gca, 'XTick', 1:length(common_residues), 'XTickLabel', common_residues, ...
         'XTickLabelRotation', 90, 'FontSize', 6);
title('RMSD Barcode: GPX4 WT (6HN3) vs GPX4–ML162 Complex (6HKQ)');
