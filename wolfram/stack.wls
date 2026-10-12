#!/usr/bin/env wolframscript
(* Stage 1 (Wolfram version of process/stack.py): cosmetic correction, registration, sigma-clip stacking.
   Raw frames *.fit in the repository root, file name "..._Bin1_<n>_..." with n = 1..7 -> L R G B S H O.

     wolframscript -file wolfram/stack.wls          (all filters)
     wolframscript -file wolfram/stack.wls 5 6 7    (only S H O)

   Output: wolfram/out/master_<F>.fits and wolfram/out/reference.txt
   Registration (one method for all filters, replaces astroalign's triangles):
     1. phase correlation of high-passed, 4x4-binned frames -> coarse translation (works with 10 stars)
     2. mutual nearest-neighbour star matching around that guess, radii 20 -> 6 -> 3 px
     3. FindGeometricTransform (similarity) + 3 MAD outlier rejection
*)
here = DirectoryName[$InputFileName];
Get[FileNameJoin[{here, "AstroTools.wl"}]];
root = ParentDirectory[here];
out = FileNameJoin[{here, "out"}];
If[!DirectoryQ[out], CreateDirectory[out]];
names = <|1 -> "L", 2 -> "R", 3 -> "G", 4 -> "B", 5 -> "S", 6 -> "H", 7 -> "O"|>;

files = FileNames["*.fit", root];
If[files === {}, Print["no *.fit in ", root]; Exit[1]];
groups = KeySort@GroupBy[files,
   ToExpression@First@StringCases[FileNameTake[#], "Bin1_" ~~ d : DigitCharacter ~~ "_" :> d] &];
only = ToExpression /@ Rest[$ScriptCommandLine];
If[only === {}, only = Keys[groups]];

(* reference = sharpest L frame *)
scores = Table[
   Module[{a = LoadFrame[f], st, fw},
    st = DetectStars[a]; fw = StarShape[a, st];
    Print["  L ", FileNameTake[f], " fwhm=", NumberForm[fw[[1]], 3], " n=", Length[st]];
    {f, fw[[1]]}],
   {f, groups[1]}];
refFile = First@First@MinimalBy[scores, Last];
ref = LoadFrame[refFile];
refXY = DetectStars[ref, 3, 1500][[All, 1 ;; 2]];   (* dense list: sparse narrowband frames still find partners *)
refHP = HighpassBin[ref];
Export[FileNameJoin[{out, "reference.txt"}], refFile, "Text"];
Print["reference: ", FileNameTake[refFile]];

Do[
  If[!MemberQ[only, k], Continue[]];
  Module[{frames = {}, valid = {}, weights = {}, name = names[k], master, rej},
   Do[
    Module[{a = LoadFrame[f], st, d, tf0, res, tf, m, warped, mask},
     st = DetectStars[a, 3, 200];
     d = PhaseShift[refHP, HighpassBin[a]];
     tf0 = TranslationTransform[{-4 d[[1]], 4 d[[2]]}];   (* array (col,row) shift -> image (x,y), y is up *)
     res = If[Length[st] >= 4, GuidedTransform[refXY, st[[All, 1 ;; 2]], tf0], $Failed];
     If[res === $Failed, Print["  ", name, " skip ", FileNameTake[f]]; Continue[]];
     tf = res[[1]]; m = TransformationMatrix[tf];
     {warped, mask} = WarpFrame[a - Median[Flatten@a[[;; ;; 8, ;; ;; 8]]], tf];
     AppendTo[frames, warped]; AppendTo[valid, mask];
     AppendTo[weights, 1/StarShape[a, st][[1]]^2];       (* sharper frames count more *)
     Print["  ", name, " ", FileNameTake[f], " shift=", NumberForm[m[[1 ;; 2, 3]], 4],
      " rot=", NumberForm[ArcTan[m[[1, 1]], m[[2, 1]]]/Degree, 3], " stars=", res[[2]],
      " rms=", NumberForm[res[[3]], 3]]],
    {f, groups[k]}];
   {master, rej} = SigmaClipStack[frames, valid, weights, 3.];
   WriteFITS[FileNameJoin[{out, "master_" <> name <> ".fits"}], master,
    <|"NCOMBINE" -> Length[frames], "FILTER" -> name|>];
   Print["== ", name, ": ", Length[frames], " frames stacked, rejected ", NumberForm[100 rej, 3], "%"]],
  {k, Keys[groups]}];
