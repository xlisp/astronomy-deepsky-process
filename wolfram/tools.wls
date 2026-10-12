#!/usr/bin/env wolframscript
(* Single-step tools (Wolfram version of scripts/*.py).

  wolframscript -file wolfram/tools.wls info      master_L.fits
  wolframscript -file wolfram/tools.wls stretch   master_L.fits L.jpg            (AutoSTF)
  wolframscript -file wolfram/tools.wls gradient  master_L.fits L_bg.fits [deg]
  wolframscript -file wolfram/tools.wls stars     master_L.fits                  (count, FWHM, ecc)
  wolframscript -file wolfram/tools.wls deconv    master_L.fits L_dc.fits [fwhm] [iters]
  wolframscript -file wolfram/tools.wls starlet   master_L.fits L_nr.fits 3,2,1,0.5
  wolframscript -file wolfram/tools.wls wavelet   master_L.fits L_nr.fits [levels]   (built-in SWT)
  wolframscript -file wolfram/tools.wls layers    master_L.fits layersdir [n]
  wolframscript -file wolfram/tools.wls starreduce image.jpg out.jpg [radius] [amount]
  wolframscript -file wolfram/tools.wls hoo       master_H.fits master_O.fits hoo.jpg
*)
Get[FileNameJoin[{DirectoryName[$InputFileName], "AstroTools.wl"}]];
args = Rest[$ScriptCommandLine];
If[Length[args] < 2, Print["usage: see header of tools.wls"]; Exit[1]];
{cmd, in} = args[[1 ;; 2]];
arg[i_, default_] := If[Length[args] >= i, ToExpression[args[[i]]], default];
load[f_] := First@ReadFITS[f];
saveAny[f_, x_] := If[StringEndsQ[f, ".fits" | ".fit"], WriteFITS[f, x], Export[f, Image[Clip[x, {0, 1}]]]];

Switch[cmd,
 "info",
  Module[{h = FITSHeader[in], a = load[in]},
   Print[KeyDrop[h, "_HeaderBytes"]];
   Print["size ", Dimensions[a], " min ", Min[a], " max ", Max[a], " median ", Median[Flatten@a[[;; ;; 4, ;; ;; 4]]],
    " noise ", NoiseSigma[a]]],
 "stretch", Export[args[[3]], Image[AutoSTF[load[in]]]],
 "gradient", WriteFITS[args[[3]], RemoveGradient[load[in], 24, arg[4, 2]]],
 "stars",
  Module[{a = load[in], st},
   st = DetectStars[a, 5, 300];
   Print[Length[st], " stars, {FWHM px, eccentricity} = ", StarShape[a, st]]],
 "deconv",
  Module[{a = load[in], fw},
   fw = arg[4, First@StarShape[a, DetectStars[a, 10, 200]]];
   Print["PSF FWHM ", fw, " px"];
   saveAny[args[[3]], Deconvolve[a, fw, arg[5, 20]]]],
 "starlet",
  Module[{a = load[in]},
   saveAny[args[[3]], StarletDenoise[a, ToExpression /@ StringSplit[args[[4]], ","], 1.]]],
 "wavelet", saveAny[args[[3]], WaveletDenoise[load[in], arg[4, 4]]],
 "layers",
  Module[{ls = StarletLayers[load[in], arg[4, 6]], dir = args[[3]]},
   If[!DirectoryQ[dir], CreateDirectory[dir]];
   MapIndexed[Export[FileNameJoin[{dir, "layer" <> ToString[First[#2]] <> ".png"}],
      ImageAdjust[Image[#1, "Real32"]]] &, ls]],
 "starreduce",
  Export[args[[3]], StarReduce[Import[in], arg[4, 1], arg[5, 0.7]]],
 "hoo",
  Module[{h = load[in], o = load[args[[3]]], hs, os, lnb},
   {hs, os} = AutoSTF[#, -2.8, 0.15] & /@ {h, o};
   lnb = AutoSTF[h/NoiseSigma[h]^2 + o/NoiseSigma[o]^2, -2.8, 0.15];
   Export[args[[4]], Image@LRGBCombine[ToRGB[{hs, 0.35 hs + 0.65 os, os}], lnb, 1.4]]],
 _, Print["unknown command ", cmd]; Exit[1]];
