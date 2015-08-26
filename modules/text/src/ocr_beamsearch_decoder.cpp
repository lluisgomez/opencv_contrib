/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"

#include <iostream>
#include <fstream>
#include <set>

namespace cv
{
namespace text
{

using namespace std;
using namespace cv::ml;

/* OCR BeamSearch Decoder */

void OCRBeamSearchDecoder::run(Mat& image, string& output_text, vector<Rect>* component_rects,
                               vector<string>* component_texts, vector<float>* component_confidences,
                               int component_level)
{
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
    output_text.clear();
    if (component_rects != NULL)
        component_rects->clear();
    if (component_texts != NULL)
        component_texts->clear();
    if (component_confidences != NULL)
        component_confidences->clear();
}
void OCRBeamSearchDecoder::run(Mat& image, Mat& mask, string& output_text, vector<Rect>* component_rects,
                               vector<string>* component_texts, vector<float>* component_confidences,
                               int component_level)
{
    CV_Assert( (image.type() == CV_8UC1) || (image.type() == CV_8UC3) );
    CV_Assert( (component_level == OCR_LEVEL_TEXTLINE) || (component_level == OCR_LEVEL_WORD) );
    output_text.clear();
    if (component_rects != NULL)
        component_rects->clear();
    if (component_texts != NULL)
        component_texts->clear();
    if (component_confidences != NULL)
        component_confidences->clear();
}


void OCRBeamSearchDecoder::ClassifierCallback::eval( InputArray image, vector< vector<double> >& recognition_probabilities, vector<int>& oversegmentation)
{
    CV_Assert(( image.getMat().type() == CV_8UC3 ) || ( image.getMat().type() == CV_8UC1 ));
    if (!recognition_probabilities.empty())
    {
        for (size_t i=0; i<recognition_probabilities.size(); i++)
            recognition_probabilities[i].clear();
    }
    recognition_probabilities.clear();
    oversegmentation.clear();
}

struct beamSearch_node {
    double score;
    vector<int> segmentation;
    bool expanded;
};

bool beam_sort_function ( beamSearch_node a, beamSearch_node b );
bool beam_sort_function ( beamSearch_node a, beamSearch_node b )
{
    return (a.score > b.score);
}


class OCRBeamSearchDecoderImpl : public OCRBeamSearchDecoder
{
public:
    //Default constructor
    OCRBeamSearchDecoderImpl( Ptr<OCRBeamSearchDecoder::ClassifierCallback> _classifier,
                              const string& _vocabulary,
                              InputArray transition_probabilities_table,
                              InputArray emission_probabilities_table,
                              decoder_mode _mode,
                              int _beam_size)
    {
        classifier = _classifier;
        emission_p = emission_probabilities_table.getMat();
        vocabulary = _vocabulary;
        mode = _mode;
        beam_size = _beam_size;
        transition_probabilities_table.getMat().copyTo(transition_p);
        for (int i=0; i<transition_p.rows; i++)
        {
            for (int j=0; j<transition_p.cols; j++)
            {
                if (transition_p.at<double>(i,j) == 0)
                    transition_p.at<double>(i,j) = -DBL_MAX;
                else
                    transition_p.at<double>(i,j) = log(transition_p.at<double>(i,j));
            }
        }
    }

    ~OCRBeamSearchDecoderImpl()
    {
    }

    void run( Mat& src,
              Mat& mask,
              string& out_sequence,
              vector<Rect>* component_rects,
              vector<string>* component_texts,
              vector<float>* component_confidences,
              int component_level)
    {
        //nothing to do with a mask here
        run( src, out_sequence, component_rects, component_texts, component_confidences, 
             component_level);
    }

    void run( Mat& src,
              string& out_sequence,
              vector<Rect>* component_rects,
              vector<string>* component_texts,
              vector<float>* component_confidences,
              int component_level)
    {

        CV_Assert( (src.type() == CV_8UC1) || (src.type() == CV_8UC3) );
        CV_Assert( (src.cols > 0) && (src.rows > 0) );
        CV_Assert( component_level == OCR_LEVEL_WORD );
        out_sequence.clear();
        if (component_rects != NULL)
            component_rects->clear();
        if (component_texts != NULL)
            component_texts->clear();
        if (component_confidences != NULL)
            component_confidences->clear();

        // TODO split a line into words

        if(src.type() == CV_8UC3)
        {
            cvtColor(src,src,COLOR_RGB2GRAY);
        }


        vector< vector<double> > recognition_probabilities;
        vector<int> oversegmentation;

        classifier->eval(src, recognition_probabilities, oversegmentation);

        // TODO if the num of oversegmentation elements is < 2 we can not do nothing!!

        int step_size = 4;  // TODO make it member of the class
        int win_size  = 32; // TODO make it member of the class

        //NMS of recognitions
        double last_best_p = 0;
        int last_best_idx  = -1;
        for (int i=0; i<recognition_probabilities.size(); )
        {
          double best_p = 0;
          int best_idx = -1;
          for (int j=0; j<recognition_probabilities[i].size(); j++)
          {
            if (recognition_probabilities[i][j] > best_p)
            {
              best_p = recognition_probabilities[i][j];
              best_idx = j;
            }
          }

          if (best_idx >=0) // this is not necessary. Here just to visualize results
          {
            cout << " " << oversegmentation[i]*step_size << "/" << vocabulary[best_idx] << " " ;
          } else {
            cout << " Err ";
          }

          if ((i>0) && (best_idx == last_best_idx) 
              && (oversegmentation[i]*step_size < oversegmentation[i-1]*step_size + win_size) )
          {
            if (last_best_p > best_p)
            {
              //remove i'th elements and do not increment i
              recognition_probabilities.erase (recognition_probabilities.begin()+i);
              oversegmentation.erase (oversegmentation.begin()+i);
              continue;
            } else {
              //remove (i-1)'th elements and do not increment i
              recognition_probabilities.erase (recognition_probabilities.begin()+i-1);
              oversegmentation.erase (oversegmentation.begin()+i-1);
              last_best_idx = best_idx;
              last_best_p   = best_p;
              continue;
            }
          }

          last_best_idx = best_idx;
          last_best_p   = best_p;
          i++;
        }
        cout << endl;


// this is not necessary. Here just to visualize results
        for (int i=0; i<recognition_probabilities.size(); )
        {
          double best_p = 0;
          int best_idx = -1;
          for (int j=0; j<recognition_probabilities[i].size(); j++)
          {
            if (recognition_probabilities[i][j] > best_p)
            {
              best_p = recognition_probabilities[i][j];
              best_idx = j;
            }
          }

          if (best_idx >=0) // this is not necessary. Here just to visualize results
          {
            cout << " " << oversegmentation[i]*step_size << "/" << vocabulary[best_idx] << " " ;
          } else {
            cout << " Err ";
          }

          i++;
        }
        cout << endl;
// up to here!


        /*Now we go here with the beam search algorithm to optimize the recognition score*/

        //convert probabilities to log probabilities
        for (size_t i=0; i<recognition_probabilities.size(); i++)
        {
            for (size_t j=0; j<recognition_probabilities[i].size(); j++)
            {
                if (recognition_probabilities[i][j] == 0)
                    recognition_probabilities[i][j] = -DBL_MAX;
                else
                    recognition_probabilities[i][j] = log(recognition_probabilities[i][j]);
            }
        }




        vector< beamSearch_node > beam;
        // Here we initialize the beam with all possible character's pairs
        int generated_chids = 0;
        for (int i=0; i<recognition_probabilities.size()-1; i++)
        {
          for (int j=i+1; j<recognition_probabilities.size(); j++)
          {

            beamSearch_node node;
            node.segmentation.push_back(i);
            node.segmentation.push_back(j);
            node.score = score_segmentation(node.segmentation, oversegmentation,
                                            recognition_probabilities, 
                                            out_sequence);
            vector< vector<int> > childs = generate_childs( node.segmentation, oversegmentation );
            node.expanded = true;
            
            beam.push_back( node );
        
            if (!childs.empty())
              update_beam( beam, oversegmentation, childs, recognition_probabilities);

            generated_chids += (int)childs.size();
            //cout << "beam size " << beam.size() << " best score " << beam[0].score<< endl;
 
          }
        }


        //cout << endl << endl << " End with initial pairs " << endl<< endl<< endl;
        //cout << "beam size " << beam.size() << " best score " << beam[0].score << endl;


        while (generated_chids != 0)
        {
            generated_chids = 0;

            for (size_t i=0; i<beam.size(); i++)
            {
                vector< vector<int> > childs;
                if (!beam[i].expanded)
                {
                  childs = generate_childs(beam[i].segmentation, oversegmentation);
                  beam[i].expanded = true;
                }
                if (!childs.empty())
                    update_beam( beam, oversegmentation, childs, recognition_probabilities);
                generated_chids += (int)childs.size();
            }
            //cout << "beam size " << beam.size() << " best score " << beam[0].score << endl;
        }


        // FINISHED ! Get the best prediction found into out_sequence
        score_segmentation(beam[0].segmentation, oversegmentation, 
                           recognition_probabilities, out_sequence);


        // TODO fill other output parameters

        return;
    }

private:

    ////////////////////////////////////////////////////////////

    // TODO the way we expand nodes makes the recognition score heuristic not monotonic
    // it should start from left node 0 and grow always to the right.

    vector< vector<int> > generate_childs(vector<int> &segmentation, vector<int> &oversegmentation)
    {

/*cout << " generate childs  for [";
for (size_t i = 0 ; i < segmentation .size(); i++)
cout << segmentation[i] << ",";
cout << "] ";*/

        vector< vector<int> > childs;
        for (size_t i=segmentation[segmentation.size()-1]+1; i<oversegmentation.size(); i++)
        {
            int seg_point = i;
            if (find(segmentation.begin(), segmentation.end(), seg_point) == segmentation.end())
            {
                //cout << seg_point << " " ;
                vector<int> child = segmentation;
                child.push_back(seg_point);
                //sort(child.begin(), child.end());
                childs.push_back(child);
            }
        }
        //cout << endl;
        return childs;
    }


    ////////////////////////////////////////////////////////////

    //TODO shall the beam itself be a member of the class?
    //     shall oversegmentation?
    void update_beam (vector< beamSearch_node > &beam, vector<int> &oversegmentation, vector< vector<int> > &childs, vector< vector<double> > &recognition_probabilities)
    {
        string out_sequence;
        double min_score = -DBL_MAX; //min score value to be part of the beam
        if ((int)beam.size() == beam_size)
            min_score = beam[beam.size()-1].score; //last element has the lowest score
        //TODO this not guaratee beam size is not increased, we must actually clamp it to 50 elements after any insert.
        for (size_t i=0; i<childs.size(); i++)
        {
            double score = score_segmentation(childs[i], oversegmentation, 
                                              recognition_probabilities, out_sequence);
            if (score > min_score)
            {
                beamSearch_node node;
                node.score = score;
                node.segmentation = childs[i];
                node.expanded = false;
                beam.push_back(node);
                sort(beam.begin(),beam.end(),beam_sort_function);
                if ((int)beam.size() > beam_size)
                {
                    beam.pop_back();
                    min_score = beam[beam.size()-1].score;
                }
            }
        }
    }


    ////////////////////////////////////////////////////////////
    // TODO Add heuristics to the score function (see PhotoOCR paper)
    // e.g.: in some cases we discard a segmentation because it includes a very large character
    //       in other cases we do it because the overlapping between two chars is too large
    //       etc.
    double score_segmentation(vector<int> &segmentation, vector<int> &oversegmentation, vector<vector<double> > &observations, string& outstring)
    {

        //cout << " start score segmentation : ";
        //for (int i=0; i<segmentation.size(); i++)
        //     cout << segmentation[i] << " ";
        //cout << endl;

        // Score Heuristics: 
        // No need to use Viterbi to know a given segmentation is bad
        // e.g.: in some cases we discard a segmentation because it includes a very large character
        //       in other cases we do it because the overlapping between two chars is too large
        int step_size = 4;  //TODO this must be memeber of the class (not hardcoded)
        int win_size  = 32; //TODO this must be memeber of the class (not hardcoded)

        Mat interdist (segmentation.size()-1, 1, CV_32F, 1);
        for (int i=0; i<segmentation.size()-1; i++)
        {
          interdist.at<float>(i,0) = oversegmentation[segmentation[i+1]]*step_size - 
                                     oversegmentation[segmentation[i]]*step_size;
          if ((float)interdist.at<float>(i,0)/win_size > 2.25) // TODO how did you set this thrs
          {
             //cout << "  rejected by aspect ratio! " << (float)interdist.at<float>(i,0)/win_size << endl;
             return -DBL_MAX;
          }
          if ((float)interdist.at<float>(i,0)/win_size < 0.15) // TODO how did you set this thrs
          {
             //cout << "  rejected by overlap! " << (float)interdist.at<float>(i,0)/win_size << endl;
             return -DBL_MAX;
          }
        }
        Scalar m, std;
        meanStdDev(interdist, m, std);
        float interdist_std = std[0];


        /*Mat overlaps (segmentation.size()+1, 1, CV_32F, 1); //we are going to penalize large variations in overlap

        for (int i=-1; i<segmentation.size(); i++)
        {
           int pairoverlap = 0;
           if (i == -1)
             pairoverlap = 0;
           else if (i == segmentation.size()-1)
             pairoverlap = 0;
           else
             pairoverlap = (segmentation[i]*step_size) + win_size - (segmentation[i+1]*step_size);

           overlaps.at<float>(i+1,0) = pairoverlap;

           if (pairoverlap > win_size/1.5) // TODO this float value is a param?
           {
             //cout << " score = 0  word = \"\"" << endl;
             cout << "  rejected by overlap! " << endl;
             return -DBL_MAX;
           }
        }

        Scalar m, std;
        meanStdDev(overlaps, m, std);
        float overlap_variance = std[0];
        if (segmentation.size() < 4)
           overlap_variance = 8;*/


        //TODO This must be extracted from dictionary
        vector<double> start_p(vocabulary.size());
        for (int i=0; i<(int)vocabulary.size(); i++)
            start_p[i] = log(1.0/vocabulary.size());


        Mat V = Mat::ones((int)segmentation.size(),(int)vocabulary.size(),CV_64FC1);
        V = V * -DBL_MAX;
        vector<string> path(vocabulary.size());

        // Initialize base cases (t == 0)
        for (int i=0; i<(int)vocabulary.size(); i++)
        {
            V.at<double>(0,i) = start_p[i] + observations[segmentation[0]][i];
            path[i] = vocabulary.at(i);
        }


        // Run Viterbi for t > 0
        for (int t=1; t<(int)segmentation.size(); t++)
        {

            vector<string> newpath(vocabulary.size());

            for (int i=0; i<(int)vocabulary.size(); i++)
            {
                double max_prob = -DBL_MAX;
                int best_idx = 0;
                for (int j=0; j<(int)vocabulary.size(); j++)
                {
                    double prob = V.at<double>(t-1,j) + transition_p.at<double>(j,i) + observations[segmentation[t]][i];
                    if ( prob > max_prob)
                    {
                        max_prob = prob;
                        best_idx = j;
                    }
                }

                V.at<double>(t,i) = max_prob;
                newpath[i] = path[best_idx] + vocabulary.at(i);
            }

            // Don't need to remember the old paths
            path.swap(newpath);
        }

        double max_prob = -DBL_MAX;
        int best_idx = 0;
        for (int i=0; i<(int)vocabulary.size(); i++)
        {
            double prob = V.at<double>((int)segmentation.size()-1,i);
            if ( prob > max_prob)
            {
                max_prob = prob;
                best_idx = i;
            }
        }

        outstring = path[best_idx];
        //cout << " score " << max_prob / (segmentation.size()-1) - overlap_variance/(step_size*segmentation.size()-1) << "(" << max_prob / (segmentation.size()-1) << " - " << overlap_variance/(step_size*segmentation.size()-1) << ") word = \"" << outstring << "\"" << endl;
        //return max_prob / (segmentation.size()-1) - overlap_variance/(step_size*segmentation.size()-1);
        //cout << " score " << max_prob / (segmentation.size()-1)  << ") word = \"" << outstring << "\"" << endl;
        return (max_prob / (segmentation.size()-1));
    }

};

Ptr<OCRBeamSearchDecoder> OCRBeamSearchDecoder::create( Ptr<OCRBeamSearchDecoder::ClassifierCallback> _classifier,
                                                        const string& _vocabulary,
                                                        InputArray transition_p,
                                                        InputArray emission_p,
                                                        decoder_mode _mode,
                                                        int _beam_size)
{
    return makePtr<OCRBeamSearchDecoderImpl>(_classifier, _vocabulary, transition_p, emission_p, _mode, _beam_size);
}


class CV_EXPORTS OCRBeamSearchClassifierCNN : public OCRBeamSearchDecoder::ClassifierCallback
{
public:
    //constructor
    OCRBeamSearchClassifierCNN(const std::string& filename);
    // Destructor
    ~OCRBeamSearchClassifierCNN() {}

    void eval( InputArray src, vector< vector<double> >& recognition_probabilities, vector<int>& oversegmentation );

protected:
    void normalizeAndZCA(Mat& patches);
    double eval_feature(Mat& feature, double* prob_estimates);

private:
    //TODO implement getters/setters for some of these members (if apply)
    int nr_class;		 // number of classes
    int nr_feature;  // number of features
    Mat feature_min; // scale range
    Mat feature_max;
    Mat weights;     // Logistic Regression weights
    Mat kernels;     // CNN kernels
    Mat M, P;        // ZCA Whitening parameters
    int step_size;   // sliding window step
    int window_size; // window size
    int quad_size;
    int patch_size;
    int num_quads;   // extract 25 quads (12x12) from each image
    int num_tiles;   // extract 25 patches (8x8) from each quad
    double alpha;    // used in non-linear activation function z = max(0, |D*a| - alpha)
};

OCRBeamSearchClassifierCNN::OCRBeamSearchClassifierCNN (const string& filename)
{
    if (ifstream(filename.c_str()))
    {
        FileStorage fs(filename, FileStorage::READ);
        // Load kernels bank and withenning params
        fs["kernels"] >> kernels;
        fs["M"] >> M;
        fs["P"] >> P;
        // Load Logistic Regression weights
        fs["weights"] >> weights;
        // Load feature scaling ranges
        fs["feature_min"] >> feature_min;
        fs["feature_max"] >> feature_max;
        fs.release();
        // TODO check all matrix dimensions match correctly and no one is empty
    }
    else
        CV_Error(Error::StsBadArg, "Default classifier data file not found!");

    nr_feature = weights.rows;
    nr_class   = weights.cols;
    // TODO some of this can be inferred from the input file (e.g. patch size must be sqrt(filters.cols))
    step_size   = 4;
    window_size = 32;
    quad_size   = 12;
    patch_size  = 8;
    num_quads   = 25;
    num_tiles   = 25;
    alpha       = 0.5;


}

void OCRBeamSearchClassifierCNN::eval( InputArray _src, vector< vector<double> >& recognition_probabilities, vector<int>& oversegmentation)
{

    CV_Assert(( _src.getMat().type() == CV_8UC3 ) || ( _src.getMat().type() == CV_8UC1 ));
    if (!recognition_probabilities.empty())
    {
        for (size_t i=0; i<recognition_probabilities.size(); i++)
            recognition_probabilities[i].clear();
    }
    recognition_probabilities.clear();
    oversegmentation.clear();


    Mat src = _src.getMat();
    if(src.type() == CV_8UC3)
    {
        cvtColor(src,src,COLOR_RGB2GRAY);
    }

    // TODO shall we resize the input image or make a copy ?
    resize(src,src,Size(window_size*src.cols/src.rows,window_size));

    int seg_points = 0;

    Mat quad;
    Mat tmp;
    Mat img;

    // begin sliding window loop foreach detection window
    for (int x_c=0; x_c<=src.cols-window_size; x_c=x_c+step_size)
    {

        img = src(Rect(Point(x_c,0),Size(window_size,window_size)));

        int patch_count = 0;
        vector< vector<double> > data_pool(9);


        int quad_id = 1;
        for (int q_x=0; q_x<=window_size-quad_size; q_x=q_x+(quad_size/2-1))
        {
            for (int q_y=0; q_y<=window_size-quad_size; q_y=q_y+(quad_size/2-1))
            {
                Rect quad_rect = Rect(q_x,q_y,quad_size,quad_size);
                quad = img(quad_rect);

                //start sliding window (8x8) in each tile and store the patch as row in data_pool
                for (int w_x=0; w_x<=quad_size-patch_size; w_x++)
                {
                    for (int w_y=0; w_y<=quad_size-patch_size; w_y++)
                    {
                        quad(Rect(w_x,w_y,patch_size,patch_size)).copyTo(tmp);
                        tmp = tmp.reshape(0,1);
                        tmp.convertTo(tmp, CV_64F);
                        normalizeAndZCA(tmp);
                        vector<double> patch;
                        tmp.copyTo(patch);
                        if ((quad_id == 1)||(quad_id == 2)||(quad_id == 6)||(quad_id == 7))
                            data_pool[0].insert(data_pool[0].end(),patch.begin(),patch.end());
                        if ((quad_id == 2)||(quad_id == 7)||(quad_id == 3)||(quad_id == 8)||(quad_id == 4)||(quad_id == 9))
                            data_pool[1].insert(data_pool[1].end(),patch.begin(),patch.end());
                        if ((quad_id == 4)||(quad_id == 9)||(quad_id == 5)||(quad_id == 10))
                            data_pool[2].insert(data_pool[2].end(),patch.begin(),patch.end());
                        if ((quad_id == 6)||(quad_id == 11)||(quad_id == 16)||(quad_id == 7)||(quad_id == 12)||(quad_id == 17))
                            data_pool[3].insert(data_pool[3].end(),patch.begin(),patch.end());
                        if ((quad_id == 7)||(quad_id == 12)||(quad_id == 17)||(quad_id == 8)||(quad_id == 13)||(quad_id == 18)||(quad_id == 9)||(quad_id == 14)||(quad_id == 19))
                            data_pool[4].insert(data_pool[4].end(),patch.begin(),patch.end());
                        if ((quad_id == 9)||(quad_id == 14)||(quad_id == 19)||(quad_id == 10)||(quad_id == 15)||(quad_id == 20))
                            data_pool[5].insert(data_pool[5].end(),patch.begin(),patch.end());
                        if ((quad_id == 16)||(quad_id == 21)||(quad_id == 17)||(quad_id == 22))
                            data_pool[6].insert(data_pool[6].end(),patch.begin(),patch.end());
                        if ((quad_id == 17)||(quad_id == 22)||(quad_id == 18)||(quad_id == 23)||(quad_id == 19)||(quad_id == 24))
                            data_pool[7].insert(data_pool[7].end(),patch.begin(),patch.end());
                        if ((quad_id == 19)||(quad_id == 24)||(quad_id == 20)||(quad_id == 25))
                            data_pool[8].insert(data_pool[8].end(),patch.begin(),patch.end());
                        patch_count++;
                    }
                }

                quad_id++;
            }
        }

        //do dot product of each normalized and whitened patch
        //each pool is averaged and this yields a representation of 9xD
        Mat feature = Mat::zeros(9,kernels.rows,CV_64FC1);
        for (int i=0; i<9; i++)
        {
            Mat pool = Mat(data_pool[i]);
            pool = pool.reshape(0,(int)data_pool[i].size()/kernels.cols);
            for (int p=0; p<pool.rows; p++)
            {
                for (int f=0; f<kernels.rows; f++)
                {
                    feature.row(i).at<double>(0,f) = feature.row(i).at<double>(0,f) + max(0.0,std::abs(pool.row(p).dot(kernels.row(f)))-alpha);
                }
            }
        }
        feature = feature.reshape(0,1);


        // data must be normalized within the range obtained during training
        double lower = -1.0;
        double upper =  1.0;
        for (int k=0; k<feature.cols; k++)
        {
            feature.at<double>(0,k) = lower + (upper-lower) *
                    (feature.at<double>(0,k)-feature_min.at<double>(0,k))/
                    (feature_max.at<double>(0,k)-feature_min.at<double>(0,k));
        }

        double *p = new double[nr_class];
        double predict_label = eval_feature(feature,p);
        //cout << " Prediction: " << vocabulary[predict_label] << " with probability " << p[0] << endl;
        if (predict_label < 0) // TODO use cvError
            cout << "OCRBeamSearchClassifierCNN::eval Error: unexpected prediction in eval_feature()" << endl;


        vector<double> recognition_p(p, p+nr_class);
        recognition_probabilities.push_back(recognition_p);
        oversegmentation.push_back(seg_points);
        seg_points++;
    }


}

// normalize for contrast and apply ZCA whitening to a set of image patches
void OCRBeamSearchClassifierCNN::normalizeAndZCA(Mat& patches)
{

    //Normalize for contrast
    for (int i=0; i<patches.rows; i++)
    {
        Scalar row_mean, row_std;
        meanStdDev(patches.row(i),row_mean,row_std);
        row_std[0] = sqrt(pow(row_std[0],2)*patches.cols/(patches.cols-1)+10);
        patches.row(i) = (patches.row(i) - row_mean[0]) / row_std[0];
    }


    //ZCA whitening
    if ((M.dims == 0) || (P.dims == 0))
    {
        Mat CC;
        calcCovarMatrix(patches,CC,M,COVAR_NORMAL|COVAR_ROWS|COVAR_SCALE);
        CC = CC * patches.rows / (patches.rows-1);


        Mat e_val,e_vec;
        eigen(CC.t(),e_val,e_vec);
        e_vec = e_vec.t();
        sqrt(1./(e_val + 0.1), e_val);


        Mat V = Mat::zeros(e_vec.rows, e_vec.cols, CV_64FC1);
        Mat D = Mat::eye(e_vec.rows, e_vec.cols, CV_64FC1);

        for (int i=0; i<e_vec.cols; i++)
        {
            e_vec.col(e_vec.cols-i-1).copyTo(V.col(i));
            D.col(i) = D.col(i) * e_val.at<double>(0,e_val.rows-i-1);
        }

        P = V * D * V.t();
    }

    for (int i=0; i<patches.rows; i++)
        patches.row(i) = patches.row(i) - M;

    patches = patches * P;

}

double OCRBeamSearchClassifierCNN::eval_feature(Mat& feature, double* prob_estimates)
{
    for(int i=0;i<nr_class;i++)
        prob_estimates[i] = 0;

    for(int idx=0; idx<nr_feature; idx++)
        for(int i=0;i<nr_class;i++)
            prob_estimates[i] += weights.at<float>(idx,i)*feature.at<double>(0,idx); //TODO use vectorized dot product

    int dec_max_idx = 0;
    for(int i=1;i<nr_class;i++)
    {
        if(prob_estimates[i] > prob_estimates[dec_max_idx])
            dec_max_idx = i;
    }

    for(int i=0;i<nr_class;i++)
        prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

    double sum=0;
    for(int i=0; i<nr_class; i++)
        sum+=prob_estimates[i];

    for(int i=0; i<nr_class; i++)
        prob_estimates[i]=prob_estimates[i]/sum;

    return dec_max_idx;
}


Ptr<OCRBeamSearchDecoder::ClassifierCallback> loadOCRBeamSearchClassifierCNN(const std::string& filename)

{
    return makePtr<OCRBeamSearchClassifierCNN>(filename);
}

}
}
