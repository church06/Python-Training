import os

import bdpy
import numpy
from bdpy.stats import corrmat

import Tools


def correct_rate(similarity):
    sample = similarity.shape[0]

    ratios = []
    for i in range(sample):
        pred = similarity[i, :]
        cr = pred[i]
        num = len(pred) - 1
        ratios.append((num - numpy.sum(pred > cr)) / float(num))

    return ratios


def correlate_correct_rate(sbj_num, t_roi, result: dict, cat_data: dict, img_feature: bdpy.BData, tools: Tools):
    cat_av_pt = cat_data['cat_pt_av']
    cat_av_im = cat_data['cat_im_av']

    layer_s = list(result[t_roi].keys())
    pred_pt = result[t_roi]
    pred_im = result[t_roi]

    ind_cat_other = (img_feature.select('FeatureType') == 4).flatten()

    nor_label = ['none', 'z-score', 'min-max', 'decimal']
    pred_label = ['p_pt_av', 'p_im_av']

    roi_dict = {}
    layer_dict = {}

    for layer, cat_pt, cat_im, p_pt, p_im in zip(layer_s, cat_av_pt, cat_av_im, pred_pt, pred_im):
        feat_other = img_feature.select(layer)[ind_cat_other, :]

        print('Layer: %s' % layer)
        feat_ca = feat_other

        feat_va_pt = numpy.vstack([cat_pt, feat_ca])
        feat_va_im = numpy.vstack([cat_im, feat_ca])

        n_sim_pt = corrmat(pred_pt[layer][nor_label[0]][pred_label[0]], feat_va_pt)
        z_sim_pt = corrmat(pred_pt[layer][nor_label[1]][pred_label[0]], feat_va_pt)
        m_sim_pt = corrmat(pred_pt[layer][nor_label[2]][pred_label[0]], feat_va_pt)
        d_sim_pt = corrmat(pred_pt[layer][nor_label[3]][pred_label[0]], feat_va_pt)

        n_sim_im = corrmat(pred_im[layer][nor_label[0]][pred_label[1]], feat_va_im)
        z_sim_im = corrmat(pred_im[layer][nor_label[1]][pred_label[1]], feat_va_im)
        m_sim_im = corrmat(pred_im[layer][nor_label[2]][pred_label[1]], feat_va_im)
        d_sim_im = corrmat(pred_im[layer][nor_label[3]][pred_label[1]], feat_va_im)

        n_cr_pt = correct_rate(n_sim_pt)
        z_cr_pt = correct_rate(z_sim_pt)
        m_cr_pt = correct_rate(m_sim_pt)
        d_cr_pt = correct_rate(d_sim_pt)

        n_cr_im = correct_rate(n_sim_im)
        z_cr_im = correct_rate(z_sim_im)
        m_cr_im = correct_rate(m_sim_im)
        d_cr_im = correct_rate(d_sim_im)

        n_cr_pt_av = numpy.mean(n_cr_pt)
        z_cr_pt_av = numpy.mean(z_cr_pt)
        m_cr_pt_av = numpy.mean(m_cr_pt)
        d_cr_pt_av = numpy.mean(d_cr_pt)

        n_cr_im_av = numpy.mean(n_cr_im)
        z_cr_im_av = numpy.mean(z_cr_im)
        m_cr_im_av = numpy.mean(m_cr_im)
        d_cr_im_av = numpy.mean(d_cr_im)

        data = {'n_cr_pt': n_cr_pt, 'z_cr_pt': z_cr_pt,
                'm_cr_pt': m_cr_pt, 'd_cr_pt': d_cr_pt,
                'n_cr_pt_av': n_cr_pt_av, 'z_cr_pt_av': z_cr_pt_av,
                'm_cr_pt_av': m_cr_pt_av, 'd_cr_pt_av': d_cr_pt_av,
                'n_cr_im': n_cr_im, 'z_cr_im': z_cr_im,
                'm_cr_im': m_cr_im, 'd_cr_im': d_cr_im,
                'n_cr_im_av': n_cr_im_av, 'z_cr_im_av': z_cr_im_av,
                'm_cr_im_av': m_cr_im_av, 'd_cr_im_av': d_cr_im_av
                }

        layer_dict[layer] = data

    roi_dict[t_roi.upper()] = layer_dict
    tools.save_final_data(sbj_num, roi_dict)


t_sbj = 's1'
t = Tools.Tool()

rs = t.read_result_data(t_sbj)
cat = t.read_cat(t_sbj)
img = bdpy.BData(os.path.abspath('data\\ImageFeatures.h5'))

for i in ['V1', 'V2', 'V3', 'V4']:
    print('ROI: %s' % i)
    correlate_correct_rate(t_sbj, i, rs, cat, img, t)
