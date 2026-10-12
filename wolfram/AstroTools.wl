(* ::Package:: *)
(* AstroTools.wl - deep-sky post-processing in the Wolfram Language.
   Wolfram counterpart of scripts/astrotools.py + process/*.py.  Load with  Get["AstroTools.wl"]

   Coordinate conventions (the #1 source of bugs when mixing array and image functions):
     * arrays   : a[[row, col]], row 1 = first FITS row (same order as astropy / numpy)
     * geometry : image coordinates {x, y}, origin at the bottom-left corner, y up, pixel (row i, col j)
                  has its centre at {j - 1/2, h - i + 1/2}. ComponentMeasurements, FindGeometricTransform
                  and ImagePerspectiveTransformation[..., DataRange -> Full] all use this system.
*)

BeginPackage["AstroTools`"];

ReadFITS::usage = "ReadFITS[file] returns {data, header}. data is a Real64 array (rows x cols), header an Association.";
FITSHeader::usage = "FITSHeader[file] parses the primary FITS header into an Association.";
WriteFITS::usage = "WriteFITS[file, data, extra:<||>] writes a 32-bit float FITS file.";
MADSigma::usage = "MADSigma[a] robust sigma = 1.4826 MedianDeviation.";
NoiseSigma::usage = "NoiseSigma[a] pixel noise from the finest scale (pixel minus mean of its 4 neighbours).";
BackgroundMap::usage = "BackgroundMap[a, tile:64] sigma-clipped tile medians, linearly resampled to full size.";
LoadFrame::usage = "LoadFrame[file] reads a FITS frame and removes hot pixels (cosmetic correction).";
DetectStars::usage = "DetectStars[a, k:5, max:300] returns {{x,y,flux}..} (image coordinates), brightest first.";
StarShape::usage = "StarShape[a, stars] returns {median FWHM px, median eccentricity} from second moments.";
HighpassBin::usage = "HighpassBin[a, b:4] high-pass + b x b max binning, input for PhaseShift.";
PhaseShift::usage = "PhaseShift[ref, img] translation {dcol, drow} with img(r) ~ ref(r - d) (phase correlation).";
GuidedTransform::usage = "GuidedTransform[refXY, xy, tf0] matches stars around a predicted transform and fits a similarity: {tf, nInliers, rms} or $Failed.";
WarpFrame::usage = "WarpFrame[a, tf] resamples frame a onto the reference grid (tf maps frame -> reference). Returns {warped, validMask}.";
SigmaClipStack::usage = "SigmaClipStack[frames, valid, weights, k:3] pixel-wise sigma-clipped weighted mean. Returns {master, rejectedFraction}.";
Deband::usage = "Deband[a] removes row/column pattern noise.";
RemoveGradient::usage = "RemoveGradient[a, grid:24, deg:2] ABE-like polynomial background subtraction.";
MTF::usage = "MTF[m, x] midtones transfer function.";
Stretch::usage = "Stretch[a, target:0.12, blackSigma:-1.5, white:Automatic] STF-like stretch to [0,1].";
AutoSTF::usage = "AutoSTF[a] PixInsight AutoSTF (shadows -2.8 MADN, target background 0.25).";
ColorBalance::usage = "ColorBalance[{r,g,b}] background neutralisation + bright-star white balance (linear).";
ToRGB::usage = "ToRGB[{r,g,b}] packs three matrices into an h x w x 3 array.";
LRGBCombine::usage = "LRGBCombine[rgb, L, sat] replaces the Lab lightness with L and scales a/b by sat.";
SCNR::usage = "SCNR[rgb, amount] average-neutral green removal.";
Denoise::usage = "Denoise[x, strength] luminance-masked bilateral smoothing of a [0,1] matrix.";
ChromaBlur::usage = "ChromaBlur[rgb, sigma] per-channel Gaussian blur.";
WorldToPixel::usage = "WorldToPixel[header, {ra, dec}] gnomonic (TAN) WCS -> 1-based FITS pixel {x, y}.";
ObjectCoordinates::usage = "ObjectCoordinates[name, fallback] {RA, Dec} in degrees from Wolfram curated data, else fallback.";
StarletLayers::usage = "StarletLayers[a, n] isotropic a-trous wavelet (B3 spline): {w1..wn, residual}.";
StarletDenoise::usage = "StarletDenoise[a, thresholds, amount:1] soft-threshold starlet layers (PixInsight MLT).";
WaveletDenoise::usage = "WaveletDenoise[a, levels:4] built-in StationaryWaveletTransform + WaveletThreshold.";
Deconvolve::usage = "Deconvolve[a, fwhm, iters:20] Richardson-Lucy with a Gaussian PSF, luminance-masked.";
StarMask::usage = "StarMask[img, k:4] soft star mask Image.";
StarReduce::usage = "StarReduce[img, radius:1, amount:0.7] erosion through the star mask.";
SaveImage::usage = "SaveImage[base, x] writes base.jpg (8 bit) and base.tif (16 bit) from a [0,1] array.";

Begin["`Private`"];

(* ------------------------------------------------------------------ FITS I/O (own parser) *)
parseValue[v_String] := Module[{s = StringTrim[v], t},
  If[StringStartsQ[s, "'"],
    Return[StringTrim@First[StringCases[s, "'" ~~ x : Shortest[___] ~~ "'" :> x], ""], Module]];
  t = StringTrim@First@StringSplit[s <> " ", "/"];
  Which[
    t === "T", True,
    t === "F", False,
    StringMatchQ[t, RegularExpression["[+-]?(\\d+\\.?\\d*|\\.\\d+)([EeDd][+-]?\\d+)?"]],
      ToExpression@StringReplace[t, {"E" | "e" | "D" | "d" -> "*^"}],
    True, t]];

FITSHeader[file_String] := Module[{cards, endp, kv},
  cards = StringPartition[StringJoin@BinaryReadList[file, "Character8", 2880*36], 80];
  endp = First@FirstPosition[cards, c_String /; StringMatchQ[c, "END" ~~ WhitespaceCharacter ...],
    {Length[cards]}, {1}];
  kv = Cases[cards[[;; endp - 1]],
    c_String /; StringTake[c, {9, 10}] === "= " :> (StringTrim@StringTake[c, 8] -> parseValue@StringDrop[c, 10])];
  Join[Association@Reverse[kv], <|"_HeaderBytes" -> 2880 Ceiling[endp 80/2880]|>]];

ReadFITS[file_String] := Module[{h = FITSHeader[file], type, dims, s, raw},
  type = Switch[h["BITPIX"], 8, "UnsignedInteger8", 16, "Integer16", 32, "Integer32",
    -32, "Real32", -64, "Real64"];
  dims = Reverse@Table[h["NAXIS" <> ToString[i]], {i, h["NAXIS"]}];   (* {[naxis3,] naxis2, naxis1} *)
  s = OpenRead[file, BinaryFormat -> True];
  SetStreamPosition[s, h["_HeaderBytes"]];
  raw = BinaryReadList[s, type, Times @@ dims, ByteOrdering -> 1];      (* FITS is big-endian *)
  Close[s];
  raw = N[raw] Lookup[h, "BSCALE", 1.] + Lookup[h, "BZERO", 0.];
  {ArrayReshape[Developer`ToPackedArray[raw], dims], h}];

fitsValue[True] = "T"; fitsValue[False] = "F";
fitsValue[v_Integer] := ToString[v];
fitsValue[v_Real] := ToString[CForm[v]];
fitsValue[v_String] := "'" <> StringPadRight[v, 8] <> "'";
card[k_String, v_String] := StringPadRight[StringPadRight[k, 8] <> "= " <> fitsValue[v], 80];
card[k_String, v_] := StringPadRight[StringPadRight[k, 8] <> "= " <> StringPadLeft[fitsValue[v], 20], 80];

WriteFITS[file_String, data_, extra_Association : <||>] := Module[{dims = Dimensions[data], hdr, s, pad},
  hdr = StringJoin@Join[
    {card["SIMPLE", True], card["BITPIX", -32], card["NAXIS", Length[dims]]},
    MapIndexed[card["NAXIS" <> ToString[First[#2]], #1] &, Reverse[dims]],
    KeyValueMap[card, KeySelect[extra, StringLength[#] <= 8 &]],
    {StringPadRight["END", 80]}];
  hdr = StringPadRight[hdr, 2880 Ceiling[StringLength[hdr]/2880]];
  s = OpenWrite[file, BinaryFormat -> True];
  BinaryWrite[s, hdr];
  BinaryWrite[s, Flatten[data], "Real32", ByteOrdering -> 1];
  pad = Mod[-4 Times @@ dims, 2880];
  If[pad > 0, BinaryWrite[s, ConstantArray[0, pad], "UnsignedInteger8"]];
  Close[s]; file];

(* ------------------------------------------------------------------ robust statistics *)
subsample[a_] := Flatten@If[ArrayDepth[a] >= 2 && Length[a] > 600, a[[;; ;; 3, ;; ;; 3]], a];
MADSigma[a_] := 1.4826 MedianDeviation[subsample[a]];
NoiseSigma[a_?MatrixQ] := MADSigma[a[[2 ;; -2, 2 ;; -2]] -
     0.25 (a[[1 ;; -3, 2 ;; -2]] + a[[3 ;;, 2 ;; -2]] + a[[2 ;; -2, 1 ;; -3]] + a[[2 ;; -2, 3 ;;]])]/Sqrt[1.25];

clippedMedian[v_List, k_ : 2.5] := Module[{m = Median[v], s = 1.4826 MedianDeviation[v], sel},
  sel = Pick[v, UnitStep[k s - Abs[v - m]], 1];
  If[sel === {}, m, Median[sel]]];

BackgroundMap[a_?MatrixQ, tile_ : 64] := Module[{h, w, gh, gw, grid},
  {h, w} = Dimensions[a]; gh = Max[1, Floor[h/tile]]; gw = Max[1, Floor[w/tile]];
  grid = Table[clippedMedian@Flatten@a[[(i - 1) tile + 1 ;; i tile, (j - 1) tile + 1 ;; j tile]],
    {i, gh}, {j, gw}];
  ArrayResample[grid, {h, w}]];

(* ------------------------------------------------------------------ frames and stars *)
LoadFrame[file_String] := Module[{a = First@ReadFITS[file], med, s},
  med = MedianFilter[a, 1];                       (* 3x3 median: what the pixel "should" be *)
  s = MADSigma[a - med];
  a + UnitStep[(a - med) - 8 s] (med - a)];       (* replace > 8 sigma hot pixels *)

DetectStars[a_?MatrixQ, k_ : 5, maxStars_ : 300] := Module[{sub, s, lab, sat, cm},
  sub = a - BackgroundMap[a];
  s = MADSigma[sub];
  lab = MorphologicalComponents[UnitStep[GaussianFilter[sub, 2] - k s/2]];   (* smoothing ~halves noise *)
  sat = 0.9 Max[sub];
  cm = ComponentMeasurements[{Image[Clip[sub, {0, Infinity}], "Real32"], lab},
    {"IntensityCentroid", "Total", "Max", "Count"}, #Count >= 3 &];
  cm = Select[Values[cm], #[[3]] < sat &];                           (* drop saturated stars *)
  Take[{#[[1, 1]], #[[1, 2]], #[[2]]} & /@ SortBy[cm, -#[[2]] &], UpTo[maxStars]]];

momentShape[sub_, {x_, y_}, r_ : 7] := Module[{h, w, i, j, p, f, xx, yy, cx, cy, mxx, myy, mxy, l1, l2},
  {h, w} = Dimensions[sub]; j = Round[x + 1/2]; i = Round[h - y + 1/2];
  If[i <= r || j <= r || i > h - r || j > w - r, Return[Nothing, Module]];
  p = Clip[sub[[i - r ;; i + r, j - r ;; j + r]], {0, Infinity}];
  f = Total[p, 2]; If[f <= 0, Return[Nothing, Module]];
  xx = Table[v, {u, -r, r}, {v, -r, r}]; yy = Transpose[xx];
  cx = Total[p xx, 2]/f; cy = Total[p yy, 2]/f;
  mxx = Total[p (xx - cx)^2, 2]/f; myy = Total[p (yy - cy)^2, 2]/f; mxy = Total[p (xx - cx) (yy - cy), 2]/f;
  {l1, l2} = Eigenvalues[{{mxx, mxy}, {mxy, myy}}];
  If[l2 <= 0.1, Nothing, {2.3548 Sqrt[(l1 + l2)/2], Sqrt[1 - l2/l1]}]];   (* l2 tiny = hot pixel *)

StarShape[a_?MatrixQ, stars_List] := Module[{sub = a - BackgroundMap[a], m},
  m = momentShape[sub, #[[1 ;; 2]]] & /@ stars;
  If[m === {}, {Infinity, 1.}, Median /@ Transpose[m]]];

(* ------------------------------------------------------------------ registration *)
HighpassBin[a_?MatrixQ, b_ : 4] := Module[{x = a - GaussianFilter[a, 12], h, w},
  {h, w} = b Floor[Dimensions[x]/b];
  BlockMap[Max, Clip[x[[;; h, ;; w]], {0, Infinity}], {b, b}]];

(* Wolfram's Fourier uses exp(+2 pi i ...) (numpy: exp(-...)); the cross-power peak still lands at +d. *)
PhaseShift[ref_?MatrixQ, img_?MatrixQ] := Module[{f, r, pos, h, w},
  f = Fourier[img] Conjugate[Fourier[ref]];
  r = Re@InverseFourier[f/(Abs[f] + 10.^-12)];
  pos = First@Position[r, Max[r]] - 1;
  {h, w} = Dimensions[r];
  {If[pos[[2]] > w/2, pos[[2]] - w, pos[[2]]], If[pos[[1]] > h/2, pos[[1]] - h, pos[[1]]]}];

GuidedTransform[refXY_List, xy_List, tf0_, radii_ : {20, 6, 3}] := Module[
  {tf = tf0, nfRef = Nearest[refXY -> "Index"], p, nfP, pairs, refP, frP, keep, res = {0.}, s, err},
  Do[
    p = tf /@ xy;
    nfP = Nearest[p -> "Index"];
    pairs = Table[{i, nfRef[p[[i]], {1, r}]}, {i, Length[p]}];
    pairs = Cases[pairs, {i_, {j_}} /; nfP[refXY[[j]], 1] === {i} :> {i, j}];   (* mutual nearest *)
    If[Length[pairs] < 4, Return[$Failed, Module]];
    refP = refXY[[pairs[[All, 2]]]]; frP = xy[[pairs[[All, 1]]]];
    keep = ConstantArray[True, Length[pairs]];
    Do[
      If[Count[keep, True] < 4, Break[]];
      {err, tf} = FindGeometricTransform[Pick[refP, keep], Pick[frP, keep],
        TransformationClass -> "Similarity"];                 (* maps frame points -> reference points *)
      res = Norm /@ ((tf /@ frP) - refP);
      s = Max[1., 3*1.4826 MedianDeviation[Pick[res, keep]]];
      keep = Thread[res < s], {3}],
    {r, radii}];
  {tf, Count[keep, True], Sqrt@Mean[Pick[res, keep]^2]}];

WarpFrame[a_?MatrixQ, tf_] := Module[{warp},
  warp[x_, rs_] := ImageData[ImagePerspectiveTransformation[Image[x, "Real32"], tf,
      DataRange -> Full, Padding -> 0., Resampling -> rs], "Real32"];
  {warp[a, "Lanczos"], UnitStep[warp[ConstantArray[1., Dimensions[a]], "Nearest"] - 0.999]}];

SigmaClipStack[frames_List, valid_List, weights_List, k_ : 3., step_ : 128] := Module[
  {h = Length[First[frames]], strips, rej = 0., tot = 0.},
  strips = Table[
    Module[{rows = r ;; Min[r + step - 1, h], cube, vm, med, sd, keep, den},
      cube = frames[[All, rows]]; vm = valid[[All, rows]];
      med = Median[cube];                                  (* element-wise median across frames *)
      sd = Clip[1.4826 Median[Abs[(# - med) & /@ cube]], {10.^-6, Infinity}];
      keep = vm (UnitStep[k sd - Abs[(# - med)]] & /@ cube);
      rej += Total[vm - keep, Infinity]; tot += Total[vm, Infinity];
      den = Total[weights keep];
      Total[weights keep cube]/Clip[den, {10.^-12, Infinity}]],
    {r, 1, h, step}];
  {Join @@ strips, rej/Max[tot, 1]}];

(* ------------------------------------------------------------------ background / banding *)
Deband[a_?MatrixQ] := Module[{x = a, med, s, prof, trend, clip},
  Do[
    med = Median[subsample[x]]; s = NoiseSigma[x];
    clip = Function[v, With[{sel = Pick[v, UnitStep[3 s - Abs[v - med]], 1]}, If[sel === {}, med, Median[sel]]]];
    prof = clip /@ If[axis == 1, x, Transpose[x]];        (* clipped median per row / per column *)
    trend = MovingAverage[ArrayPad[prof, 50, "Fixed"], 101];
    prof = prof - trend;                                  (* keep the real large-scale signal *)
    x = If[axis == 1, x - prof, (# - prof) & /@ x],
    {axis, {1, 2}}];
  x];

RemoveGradient[a_?MatrixQ, grid_ : 24, deg_ : 2] := Module[
  {h, w, gy, gx, pts, vals, exps, design, keep, c, r, s, xs, ys, coarse},
  {h, w} = Dimensions[a]; gy = Floor[h/grid]; gx = Floor[w/grid];
  pts = N@Flatten[Table[{(j - 1/2) gx/w, (i - 1/2) gy/h}, {i, grid}, {j, grid}], 1];
  vals = Flatten@Table[Median@Flatten@a[[(i - 1) gy + 1 ;; i gy, (j - 1) gx + 1 ;; j gx]], {i, grid}, {j, grid}];
  exps = Flatten[Table[{p, q}, {p, 0, deg}, {q, 0, deg - p}], 1];
  design[{x_, y_}] := x^exps[[All, 1]] y^exps[[All, 2]];   (* coordinates are > 0, so no 0^0 *)
  keep = ConstantArray[True, Length[vals]];
  Do[
    c = LeastSquares[design /@ Pick[pts, keep], Pick[vals, keep]];
    r = vals - (design /@ pts).c;
    s = 1.4826 MedianDeviation[Pick[r, keep]];
    keep = Thread[r < 2.5 s],                              (* drop tiles brighter than the model *)
    {5}];
  xs = (Range[Ceiling[w/8]] - 1/2)/Ceiling[w/8]; ys = (Range[Ceiling[h/8]] - 1/2)/Ceiling[h/8];
  coarse = Table[design[{x, y}].c, {y, N@ys}, {x, N@xs}];
  With[{out = a - ArrayResample[coarse, {h, w}]}, out - Median[subsample[out]]]];

(* ------------------------------------------------------------------ stretch / colour *)
MTF[m_, x_] := ((m - 1) x)/((2 m - 1) x - m);

Stretch[a_, target_ : 0.12, blackSigma_ : -1.5, white_ : Automatic] := Module[
  {med = Median[subsample[a]], s = NoiseSigma[a], lo, hi, b},
  lo = med + blackSigma s;
  hi = If[white === Automatic, Quantile[subsample[a], 0.99995], white];
  b = (med - lo)/(hi - lo);
  MTF[MTF[target, b], Clip[(a - lo)/(hi - lo), {0, 1}]]];  (* MTF[MTF[t,b],b] == t *)

AutoSTF[a_?MatrixQ, shadows_ : -2.8, target_ : 0.25] := Module[{x, med, c0},
  x = Rescale[a]; med = Median[subsample[x]];
  c0 = Clip[med + shadows MADSigma[x], {0, 1}];
  MTF[MTF[target, (med - c0)/(1 - c0)], Clip[(x - c0)/(1 - c0), {0, 1}]]];

ToRGB[chs_List] := Transpose[chs, {3, 1, 2}];

ColorBalance[chs_List] := Module[{out, f},
  out = (# - Median[subsample[#]]) & /@ chs;
  f = Function[c, With[{v = subsample[c]}, With[{t = Quantile[v, 0.999], mx = Max[v]},
      Median@Pick[v, UnitStep[v - t] UnitStep[0.8 mx - v], 1]]]] /@ out;
  MapThread[#1 First[f]/#2 &, {out, f}]];

LRGBCombine[rgb_, l_?MatrixQ, sat_ : 1.3] := Module[{lab},
  lab = ImageData@ColorConvert[Image[Clip[rgb, {0, 1}], ColorSpace -> "RGB"], "LAB"];
  lab[[All, All, 1]] = l;
  lab[[All, All, 2 ;; 3]] *= sat;
  Clip[ImageData@ColorConvert[Image[lab, ColorSpace -> "LAB"], "RGB"], {0, 1}]];

SCNR[rgb_, amount_ : 1.] := Module[{r = rgb[[All, All, 1]], g = rgb[[All, All, 2]], b = rgb[[All, All, 3]]},
  ToRGB[{r, (1 - amount) g + amount (g - Ramp[g - (r + b)/2]), b}]];   (* g - Ramp[g-lim] == Min[g, lim] *)

ChromaBlur[rgb_, sigma_] := ImageData@GaussianFilter[Image[rgb], {Ceiling[3 sigma], sigma}];

Denoise[x_?MatrixQ, strength_ : 0.1] := Module[{blur, w},
  blur = ImageData@BilateralFilter[Image[x, "Real32"], 3, strength];
  w = Clip[x/0.5, {0, 1}];                                (* bright areas keep their detail *)
  w x + (1 - w) blur];

(* ------------------------------------------------------------------ astrometry *)
WorldToPixel[h_Association, {ra_, dec_}] := Module[{a0, d0, a, d, cd, cosc, xi, eta},
  {a0, d0} = {h["CRVAL1"], h["CRVAL2"]} Degree; {a, d} = {ra, dec} Degree;
  cd = If[KeyExistsQ[h, "CD1_1"],
    {{h["CD1_1"], Lookup[h, "CD1_2", 0.]}, {Lookup[h, "CD2_1", 0.], h["CD2_2"]}},
    With[{t = Lookup[h, "CROTA2", 0.] Degree, c1 = h["CDELT1"], c2 = h["CDELT2"]},
      {{c1 Cos[t], -c2 Sin[t]}, {c1 Sin[t], c2 Cos[t]}}]];
  cosc = Sin[d0] Sin[d] + Cos[d0] Cos[d] Cos[a - a0];
  xi = Cos[d] Sin[a - a0]/cosc/Degree;
  eta = (Cos[d0] Sin[d] - Sin[d0] Cos[d] Cos[a - a0])/cosc/Degree;
  {h["CRPIX1"], h["CRPIX2"]} + Inverse[cd].{xi, eta}];

ObjectCoordinates[name_String, fallback_List] := Module[{v},
  v = Quiet@Check[QuantityMagnitude@UnitConvert[
       {NebulaData[name, "RightAscension"], NebulaData[name, "Declination"]}, "AngularDegrees"], $Failed];
  If[VectorQ[v, NumericQ] && Length[v] == 2, N[v], fallback]];

(* ------------------------------------------------------------------ wavelets *)
b3 = {1., 4., 6., 4., 1.}/16;
holed[j_] := Module[{s = 2^j, k}, k = ConstantArray[0., 4 s + 1]; k[[1 ;; ;; s]] = b3; k];
sepSmooth[c_, k_] := Module[{r = (Length[k] - 1)/2},
  ListConvolve[Transpose[{k}], ArrayPad[ListConvolve[{k}, ArrayPad[c, {{0, 0}, {r, r}}, "Reflected"]],
    {{r, r}, {0, 0}}, "Reflected"]]];

StarletLayers[a_?MatrixQ, n_Integer] := Module[{c = a, nxt, layers = {}},
  Do[nxt = sepSmooth[c, holed[j]]; AppendTo[layers, c - nxt]; c = nxt, {j, 0, n - 1}];
  Append[layers, c]];                                     (* Total[layers] == a *)

starletNoise = {0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005, 0.0025};

StarletDenoise[a_?MatrixQ, k_List, amount_ : 1.] := Module[{s = NoiseSigma[a], layers},
  layers = StarletLayers[a, Length[k] + 1];
  Total@MapIndexed[
    Function[{w, i}, If[First[i] <= Length[k] && k[[First[i]]] > 0,
      With[{t = k[[First[i]]] s starletNoise[[First[i]]]},
        (1 - amount) w + amount Sign[w] Ramp[Abs[w] - t]], w]], layers]];

WaveletDenoise[a_?MatrixQ, levels_ : 4] := Module[{lo = Min[a], hi = Max[a], dwd},
  dwd = StationaryWaveletTransform[Image[(a - lo)/(hi - lo), "Real32"], DaubechiesWavelet[4], levels];
  ImageData[InverseWaveletTransform[WaveletThreshold[dwd, {"Soft", "Universal"}]], "Real32"] (hi - lo) + lo];

(* ------------------------------------------------------------------ deconvolution / stars *)
Deconvolve[a_?MatrixQ, fwhm_, iters_ : 20, dering_ : 0.5] := Module[
  {lo = Min[a], hi = Max[a], x, sigma = fwhm/2.3548, dec, lmask, bright, w},
  x = (a - lo)/(hi - lo);
  dec = ImageData[ImageDeconvolve[Image[x, "Real32"], GaussianMatrix[{Ceiling[4 sigma], sigma}],
     Method -> "RichardsonLucy", MaxIterations -> iters], "Real32"];
  lmask = Clip[(AutoSTF[a, -2.8, 0.15] - 0.2)/0.3, {0, 1}];        (* background 0, signal 1 *)
  bright = ImageData@Blur[Dilation[Image[N@UnitStep[x - dering], "Real32"], Ceiling[2 fwhm]], Ceiling[fwhm/2]];
  w = lmask (1 - Clip[bright, {0, 1}]);                            (* deringing: skip the brightest stars *)
  (w dec + (1 - w) x) (hi - lo) + lo];

StarMask[img_Image, k_ : 4] := Module[{lum = ColorConvert[img, "Grayscale"], small},
  small = ImageSubtract[lum, GaussianFilter[lum, 8]];              (* remove nebula / large scales *)
  Blur[Dilation[DeleteSmallComponents[Binarize[small, k NoiseSigma[ImageData[lum]]], 3], 2], 4]];

StarReduce[img_Image, radius_ : 1, amount_ : 0.7] := Module[{d = ImageData[img], er, m},
  er = ImageData@Erosion[img, DiskMatrix[radius]];
  m = amount ImageData[StarMask[img]];
  If[ArrayDepth[d] == 3, m = ToRGB[{m, m, m}]];
  Image[(1 - m) d + m (d - Ramp[d - er])]];

SaveImage[base_String, x_] := (
  Export[base <> ".jpg", Image[Clip[x, {0, 1}]], "CompressionLevel" -> 0.05];
  Export[base <> ".tif", Image[Clip[x, {0, 1}], "Bit16"]];
  Print["saved ", base]);

End[];
EndPackage[];
