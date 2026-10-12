#!/usr/bin/env wolframscript
(* Stage 2 (Wolfram version of process/compose.py): deband, gradient removal, WCS crop, stretch,
   LRGB / SHO / HOO / HaLRGB composition.

     wolframscript -file wolfram/compose.wls            (masters from wolfram/out)
     wolframscript -file wolfram/compose.wls python     (masters made by process/stack.py)
*)
here = DirectoryName[$InputFileName];
Get[FileNameJoin[{here, "AstroTools.wl"}]];
root = ParentDirectory[here];
out = FileNameJoin[{here, "out"}];
src = If[MemberQ[Rest[$ScriptCommandLine], "python"], FileNameJoin[{root, "process"}], out];

m = AssociationMap[First@ReadFITS[FileNameJoin[{src, "master_" <> # <> ".fits"}]] &,
   {"L", "R", "G", "B", "S", "H", "O"}];

(* common valid area: drift leaves zero borders *)
valid = Times @@ (Unitize /@ Values[m]);
{rows, cols} = {Flatten@Position[Total[valid, {2}], _?Positive], Flatten@Position[Total[valid], _?Positive]};
{y0, y1, x0, x1} = {Min[rows] + 20, Max[rows] - 20, Min[cols] + 20, Max[cols] - 20};
m = #[[y0 ;; y1, x0 ;; x1]] & /@ m;

m = RemoveGradient@*Deband /@ m;
KeyValueMap[Print[#1, " noise ", NoiseSigma[#2]] &, m];

(* locate M27 through the plate solution (TAN projection, implemented in AstroTools) *)
refFile = Quiet@Check[StringTrim@Import[FileNameJoin[{out, "reference.txt"}], "Text"], $Failed];
If[!StringQ[refFile] || !FileExistsQ[refFile], refFile = First@FileNames["*Bin1_1_*_0006.fit", root]];
radec = ObjectCoordinates["M27", {299.9016, 22.7212}];
{px, py} = WorldToPixel[FITSHeader[refFile], radec];           (* 1-based FITS x (column), y (row) *)
half = 1000;
{h, w} = Dimensions[m["H"]];
cy = Clip[Round[py] - y0 + 1, {half + 1, h - half}];
cx = Clip[Round[px] - x0 + 1, {half + 1, w - half}];
Print["M27 ", radec, " -> centre (col,row) ", {cx, cy}];
c = #[[cy - half ;; cy + half - 1, cx - half ;; cx + half - 1]] & /@ m;

(* ---------- LRGB ---------- *)
{r, g, b} = ColorBalance[{c["R"], c["G"], c["B"]}];
white = Quantile[Flatten[{r[[;; ;; 4, ;; ;; 4]], g[[;; ;; 4, ;; ;; 4]], b[[;; ;; 4, ;; ;; 4]]}], 0.9999];
rgbS = ToRGB[Stretch[#, 0.10, -1.5, white] & /@ {r, g, b}];
lsyn = c["L"]/NoiseSigma[c["L"]]^2 + (r + g + b)/NoiseSigma[r + g + b]^2;   (* inverse-variance weights *)
lS = Denoise[Stretch[lsyn, 0.10], 0.1];
rgbS = ChromaBlur[rgbS, 1.5];
lrgb = SCNR[LRGBCombine[rgbS, lS, 1.6], 0.8];
SaveImage[FileNameJoin[{out, "M27_LRGB"}], lrgb];

(* ---------- narrowband ---------- *)
nbWhite = Quantile[Flatten@#[[;; ;; 4, ;; ;; 4]], 0.9998] &;
{sS, hS, oS} = Denoise[Stretch[#, 0.08, -1.5, nbWhite[#]], 0.12] & /@ {c["S"], c["H"], c["O"]};
lnb = Denoise[Stretch[c["H"]/NoiseSigma[c["H"]]^2 + c["O"]/NoiseSigma[c["O"]]^2 +
      c["S"]/NoiseSigma[c["S"]]^2, 0.10], 0.1];

sho = SCNR[ToRGB[{0.55 sS + 0.45 hS, 0.6 hS + 0.4 oS, oS}], 0.3];        (* Hubble palette *)
SaveImage[FileNameJoin[{out, "M27_SHO"}], LRGBCombine[ChromaBlur[sho, 1.2], lnb, 1.5]];

hoo = ToRGB[{hS, 0.35 hS + 0.65 oS, oS}];
SaveImage[FileNameJoin[{out, "M27_HOO"}], LRGBCombine[ChromaBlur[hoo, 1.2], lnb, 1.4]];

(* LRGB enhanced with Ha (red) and OIII (green/blue); Max via a + Ramp[b - a] *)
hn = Stretch[c["H"], 0.06, -1.5, nbWhite[c["H"]]];
on = Stretch[c["O"], 0.06, -1.5, nbWhite[c["O"]]];
enh = ToRGB[{#1 + Ramp[hn - #1], #2 + Ramp[0.5 on - #2], #3 + Ramp[on - #3]} & @@
    Transpose[rgbS, {2, 3, 1}]];
lEnh = lS + Ramp[0.6 lnb - lS];
SaveImage[FileNameJoin[{out, "M27_HaLRGB"}], SCNR[LRGBCombine[ChromaBlur[enh, 1.5], lEnh, 1.5], 0.7]];

SaveImage[FileNameJoin[{out, "M27_L_mono"}], lS];
MapThread[SaveImage[FileNameJoin[{out, "M27_" <> #1 <> "_mono"}], #2] &, {{"S", "H", "O"}, {sS, hS, oS}}];
