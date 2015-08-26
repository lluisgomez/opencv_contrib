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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/datasets/tr_icdar.hpp"

#include <opencv2/core.hpp>

#include "opencv2/text.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <cstdio>
#include <cstdlib> // atoi

#include <iostream>
#include <fstream>

#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace cv::text;

//Calculate edit distance between two words
size_t edit_distance(const string& A, const string& B);
size_t min(size_t x, size_t y, size_t z);

size_t min(size_t x, size_t y, size_t z)
{
    return x < y ? min(x,z) : min(y,z);
}

size_t edit_distance(const string& A, const string& B)
{
    size_t NA = A.size();
    size_t NB = B.size();

    vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));

    for (size_t a = 0; a <= NA; ++a)
        M[a][0] = a;

    for (size_t b = 0; b <= NB; ++b)
        M[0][b] = b;

    for (size_t a = 1; a <= NA; ++a)
        for (size_t b = 1; b <= NB; ++b)
        {
            size_t x = M[a-1][b] + 1;
            size_t y = M[a][b-1] + 1;
            size_t z = M[a-1][b-1] + (A[a-1] == B[b-1] ? 0 : 1);
            M[a][b] = min(x,y,z);
        }

    return M[A.size()][B.size()];
}


int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset root folder }"
            "{ ws wordspotting|    | evaluate \"word spotting\" results }"
            "{ lex lexicon    |1   | 0:no-lexicon, 1:100-words, 2:full-lexicon }";

    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    bool is_word_spotting = parser.has("ws");
    int selected_lex = parser.get<int>("lex");
    if ((selected_lex < 0) || (selected_lex > 2))
    {
        parser.printMessage();
        printf("Unsupported lex value.\n");
        return -1;
    }

    //load lexicon
    vector<string> old_lexicon;
    string word;
    ifstream infile("Lex_test.txt");
    while (infile >> word)
      old_lexicon.push_back(word);
    infile.close();
    

    // loading train & test images description
    Ptr<TR_icdar> dataset = TR_icdar::create();
    dataset->load(path);

    unsigned int correctNum = 0;
    unsigned int returnedCorrectNum = 0;

    vector< Ptr<Object> >& test = dataset->getTest();
    unsigned int num = 0;
    for (vector< Ptr<Object> >::iterator itT=test.begin(); itT!=test.end(); ++itT)
    {
        TR_icdarObj *example = static_cast<TR_icdarObj *>((*itT).get());

        num++;
        printf("processed image: %u, name: %s\n", num, example->fileName.c_str());

        Mat image = imread((path+"/test/"+example->fileName).c_str());
        Mat grey;
        cvtColor(image,grey,COLOR_RGB2GRAY);

        // create an OCRBeamSearchDecoder instance w./w.o. lexicon
        string vocabulary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"; // must have the same order as the clasifier output classes
        Mat transition_p;
        vector<string> tailored_lexicon;
        vector<string> *lex;
        switch (selected_lex)
        {
            case 0: // no lexicon (use a generic language model)
            {
                lex = &tailored_lexicon; // (empty lexicon)
                string filename = "OCRHMM_transitions_table.xml"; 
                FileStorage fst(filename, FileStorage::READ);
                fst["transition_probabilities"] >> transition_p;
                fst.release();
                break;
            }
            case 2: // ICDAR Full lexicon
            {
                /*//ICDAR lexicon in only uppercase but words may be lowercase/title
                for (size_t l=0; l<example->lexFull.size(); l++)
                {
                    tailored_lexicon.push_back(example->lexFull[l]);
                    for (size_t c=1; c<example->lexFull[l].size(); c++)
                    {
                        example->lexFull[l][c] = tolower(example->lexFull[l][c]);
                    }
                    tailored_lexicon.push_back(example->lexFull[l]);
                    example->lexFull[l][0] = tolower(example->lexFull[l][0]);
                    tailored_lexicon.push_back(example->lexFull[l]);
                }
                lex = &tailored_lexicon;*/
                lex = &old_lexicon;
                createOCRHMMTransitionsTable(vocabulary,tailored_lexicon,transition_p);
                break;
            }
            default: // ICDAR-100 lexicon (100 words)
            {
                /*for (size_t l=0; l<example->lex100.size(); l++)
                {
                    tailored_lexicon.push_back(example->lex100[l]);
                    for (size_t c=1; c<example->lex100[l].size(); c++)
                    {
                        example->lex100[l][c] = tolower(example->lex100[l][c]);
                    }
                    tailored_lexicon.push_back(example->lex100[l]);
                    example->lex100[l][0] = tolower(example->lex100[l][0]);
                    tailored_lexicon.push_back(example->lex100[l]);
                }
                lex = &tailored_lexicon;*/
                for (size_t w=0; w<example->words.size(); w++)
                {
                    string alnum_value = example->words[w].value;
                    for (size_t c=0; c<alnum_value.size(); c++)
                    {
                      if (!isalnum(alnum_value[c]))
                      {
                        alnum_value = alnum_value.substr(0,c);
                        break;
                      }
                    }
                    if (alnum_value.size()>2)
                      tailored_lexicon.push_back(alnum_value);
                }
                for (size_t w=example->words.size(); w<50; w++)
                {
                   tailored_lexicon.push_back(old_lexicon[rand() % old_lexicon.size()]);
                }
                lex = &tailored_lexicon;
                createOCRHMMTransitionsTable(vocabulary,tailored_lexicon,transition_p);
                break;
            }
        }
    
        Mat emission_p = Mat::eye(62,62,CV_64FC1);
    
        Ptr<OCRBeamSearchDecoder> ocr = OCRBeamSearchDecoder::create(
                    loadOCRBeamSearchClassifierCNN("OCRBeamSearch_CNN_model_data.xml.gz"),
                    vocabulary, transition_p, emission_p);



        for (size_t w=0; w<example->words.size(); w++)
        {
            string w_upper = example->words[w].value;
            cout << "GT transcription = \"" << w_upper << "\"" << endl;
            // ICDAR protocol accepts also recognition up to the first non alphanumeric char
            string alnum_value = w_upper;
            for (size_t c=0; c<alnum_value.size(); c++)
            {
                if (!isalnum(alnum_value[c]))
                {
                    alnum_value = alnum_value.substr(0,c);
                    break;
                }
            }

            // If in word_spotting scenario we do not care about words that are not in lexicon
            if ((find (lex->begin(), lex->end(), w_upper) == lex->end()) &&
                (is_word_spotting) && (selected_lex != 0))
            {
                continue;
            }

            // Take care of dontcare regions (t.value == "###")
            if ( (example->words[w].value == "###") || (example->words[w].value.size()<3) )
            {
                continue;
            } 
            else 
            {
                correctNum ++;
                /*Text Recognition (OCR)*/
                string output;
                vector<Rect>   boxes;
                vector<string> words;
                vector<float>  confidences;
                Rect word_rect = Rect(example->words[w].x, example->words[w].y,
                                      example->words[w].width, example->words[w].height);
                // add some space at left/right borders
                word_rect.x = word_rect.x - word_rect.height/2;
                word_rect.width = word_rect.width + word_rect.height;
                word_rect = word_rect & Rect(0,0,grey.cols,grey.rows);
                Mat crop;
                grey(Rect(word_rect)).copyTo(crop);
                ocr->run(crop, output, &boxes, &words, &confidences, OCR_LEVEL_WORD);

                if (output.size() < 3) continue;
        
                cout << "OCR output = \"" << output << "\"" << endl;
        
                /* Predicted words which are not in the lexicon are filtered
                   or changed to match one (when edit distance ratio < 0.34)*/
                float max_edit_distance_ratio = (float)0.34;

                if (lex->size() > 0)
                {
                    if (find(lex->begin(), lex->end(), output) == lex->end())
                    {
                        int best_match = -1;
                        int best_dist  = output.size();
                        for (size_t l=0; l<lex->size(); l++)
                        {
                            int dist = edit_distance(lex->at(l),output);
                            if (dist < best_dist)
                            {
                                best_match = l;
                                best_dist = dist;
                            }
                        }
                        if (best_dist/output.size() < max_edit_distance_ratio)
                            output = lex->at(best_match);
                        else
                            continue;
                    }
                }
    
                if ((find (lex->begin(), lex->end(), output)
                     == lex->end()) && (is_word_spotting) && (selected_lex != 0))
                   continue;

                transform(w_upper.begin(), w_upper.end(), w_upper.begin(), ::toupper);
                transform(output.begin(), output.end(), output.begin(), ::toupper);
                cout << "GT transcription = \"" << w_upper << "\"" << endl;
                cout << "OCR output       = \"" << output << "\"" << endl;
                if ( (w_upper==output) || (alnum_value==output) )
                {
                    returnedCorrectNum++;
                    cout << "OK!" << endl;
                }
                else cout << "FAIL." << endl;

                printf(" (Partial)   Accuracy: %d/%d = %f\n", returnedCorrectNum, correctNum, 1.0*returnedCorrectNum/correctNum);
            
                /*namedWindow("crop",WINDOW_NORMAL);
                imshow("crop",crop);
                imwrite("crop.jpg", crop);
                waitKey(-1);*/
            }
        }
    }

    double accuracy = 1.0*returnedCorrectNum/correctNum;

    printf("\n-------------------------------------------------------------------------\n");
    printf("ICDAR2015 -- Challenge 2: \"Focused Scene Text\" -- Task 3 \"Word Recognition\"\n");
    if (is_word_spotting) printf("             Word spotting results -- ");
    else printf("             Recognition results -- ");
    switch (selected_lex)
    {
        case 0:
            printf("generic recognition (no given lexicon)\n");
            break;
        case 2:
            printf("weakly contextualized lexicon (624 words)\n");
            break;
        default:
            printf("strongly contextualized lexicon (100 words)\n");
            break;
    }
    printf("             Accuracy: %f\n", accuracy);
    printf("-------------------------------------------------------------------------\n\n");

    return 0;
}
