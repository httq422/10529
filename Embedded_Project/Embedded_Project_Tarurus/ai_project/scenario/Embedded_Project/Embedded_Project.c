#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/prctl.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "posix_help.h"
#include "audio_aac_adp.h"
#include "base_interface.h"
#include "osd_img.h"
#include "Embedded_Project.h"
#include "yolov2_hand_detect.h"
#include "misc_util.h"
#include "hisignalling.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define MODEL_FILE_TRASH    "/userdata/models/Embedded_Project/PlayPhone4_inst.wk" 
#define SCORE_MAX           4096 
#define DETECT_OBJ_MAX      32
#define RET_NUM_MAX         4
#define THRESH_MIN          30    

#define FRM_WIDTH           256
#define FRM_HEIGHT          256
#define TXT_BEGX            20
#define TXT_BEGY            20

static int g_num = 108;
static int g_count = 0;
#define AUDIO_CASE_TWO     2
#define AUDIO_SCORE        40       // Confidence can be configured by yourself
#define AUDIO_FRAME        14       // Recognize once every 15 frames, can be configured by yourself

#define MULTIPLE_OF_EXPANSION 100   // Multiple of expansion
#define UNKOWN_WASTE          20    // Unkown Waste
#define BUFFER_SIZE           16    // buffer size
#define MIN_OF_BOX            16    // min of box
#define MAX_OF_BOX            240   // max of box

static HI_BOOL g_bAudioProcessStopSignal = HI_FALSE;
static pthread_t g_audioProcessThread = 0;
static OsdSet* g_osdsTrash = NULL;
static HI_S32 g_osd0Trash = -1;
int uartFd = 0;

static SkPair g_stmChn = {
    .in = -1,
    .out = -1
};

/*
 * 将识别的结果进行音频播放
 * Audio playback of the recognition results
 */
static HI_VOID PlayAudio(const RecogNumInfo items)
{
    if  (g_count < AUDIO_FRAME) {
        g_count++;
        return;
    }

    const RecogNumInfo *item = &items;
    uint32_t score = item->score * MULTIPLE_OF_EXPANSION / SCORE_MAX;
    if ((score > AUDIO_SCORE) && (g_num != item->num)) {
        g_num = item->num;
        if (g_num != UNKOWN_WASTE) {
            AudioTest(g_num, -1);
        }
    }
    g_count = 0;
}

static HI_VOID* GetAudioFileName(HI_VOID* arg)
{
    RecogNumInfo resBuf = {0};
    int ret;

    while (g_bAudioProcessStopSignal == false) {
        ret = FdReadMsg(g_stmChn.in, &resBuf, sizeof(RecogNumInfo));
        if (ret == sizeof(RecogNumInfo)) {
            PlayAudio(resBuf);
        }
    }

    return NULL;
}

/*
 * 加载姿势分类wk模型
 * Load the classification wk model
 */
HI_S32 CnnTrashClassifyLoadModel(uintptr_t* model, OsdSet* osds)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;
    HI_CHAR audioThreadName[BUFFER_SIZE] = {0};

    ret = OsdLibInit();
    HI_ASSERT(ret == HI_SUCCESS);

    g_osdsTrash = osds;
    HI_ASSERT(g_osdsTrash);
    g_osd0Trash = OsdsCreateRgn(g_osdsTrash);
    HI_ASSERT(g_osd0Trash >= 0);

    ret = CnnCreate(&self, MODEL_FILE_TRASH);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("load cnn manner classify model, ret:%d\n", ret);

    if (GetCfgBool("audio_player:support_audio", true)) {
        ret = SkPairCreate(&g_stmChn);
        HI_ASSERT(ret == 0);
        if (snprintf_s(audioThreadName, BUFFER_SIZE, BUFFER_SIZE - 1, "AudioProcess") < 0) {
            HI_ASSERT(0);
        }
        prctl(PR_SET_NAME, (unsigned long)audioThreadName, 0, 0, 0);
        ret = pthread_create(&g_audioProcessThread, NULL, GetAudioFileName, NULL);
        if (ret != 0) {
            SAMPLE_PRT("audio proccess thread creat fail:%s\n", strerror(ret));
            return ret;
        }
    }

    /*
     * Uart串口初始化
     * Uart open init
     */
    uartFd = UartOpenInit();
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}

/*
 * 卸载姿势分类wk模型
 * Unload the classification wk model
 */
HI_S32 CnnTrashClassifyUnloadModel(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    SAMPLE_PRT("unload manner classify model success\n");
    OsdsClear(g_osdsTrash);

    close(uartFd);
    if (GetCfgBool("audio_player:support_audio", true)) {
        SkPairDestroy(&g_stmChn);
        SAMPLE_PRT("SkPairDestroy success\n");
        g_bAudioProcessStopSignal = HI_TRUE;
        pthread_join(g_audioProcessThread, NULL);
        g_audioProcessThread = 0;
    }

    return HI_SUCCESS;
}

/*
 * 根据推理结果进行业务处理
 * Perform business processing based on inference results
 */
static HI_S32 CnnTrashClassifyFlag(const RecogNumInfo items[], HI_S32 itemNum, HI_CHAR* buf, HI_S32 size)
{
    HI_S32 offset = 0;
    HI_CHAR *mannerName = NULL;

    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, "manner classify: {");
    for (HI_U32 i = 0; i < itemNum; i++) {
        const RecogNumInfo *item = &items[i];
        uint32_t score = item->score * HI_PER_BASE / SCORE_MAX;
        if (score < THRESH_MIN) {
            break;
        }
        SAMPLE_PRT("----manner flag----num:%d, score:%d\n", item->num, score);
        switch (item->num) {
            case 0u:
                mannerName = "GoodManner";
                SAMPLE_PRT("----Detection----:%s\n", mannerName);
				UartSendRead(uartFd, InvalidGesture);
                break;
            default:
                mannerName = "PlayPhone";
            SAMPLE_PRT("----Detection----:%s\n", mannerName);
            UartSendRead(uartFd, FistGesture);
            break;
                
        }
        offset += snprintf_s(buf + offset, size - offset, size - offset - 1,
            "%s%s %u:%u%%", (i == 0 ? " " : ", "), mannerName, (int)item->num, (int)score);
        HI_ASSERT(offset < size);
    }
    offset += snprintf_s(buf + offset, size - offset, size - offset - 1, " }");
    HI_ASSERT(offset < size);
    return HI_SUCCESS;
}

/*
 * 先进行预处理，再使用NNIE进行硬件加速推理，不支持层通过AI CPU进行计算
 *
 * Perform preprocessing first, and then use NNIE for hardware accelerated inference,
 * and do not support layers to be calculated by AI CPU
 */
HI_S32 CnnTrashClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *resFrm)
{
    SAMPLE_PRT("begin CnnTrashClassifyCal\n");
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model; // reference to SDK sample_comm_nnie.h Line 99
    IVE_IMAGE_S img; // referece to SDK hi_comm_ive.h Line 143
    RectBox cnnBoxs[DETECT_OBJ_MAX] = {0};
    VIDEO_FRAME_INFO_S resizeFrm;  // Meet the input frame of the plug
    static HI_CHAR prevOsd[NORM_BUF_SIZE] = "";
    HI_CHAR osdBuf[NORM_BUF_SIZE] = "";
    RecogNumInfo resBuf[RET_NUM_MAX] = {0};
    HI_S32 resLen = 0;
    HI_S32 ret;
    IVE_IMAGE_S imgIn;

    cnnBoxs[0].xmin = MIN_OF_BOX;
    cnnBoxs[0].xmax = MAX_OF_BOX;
    cnnBoxs[0].ymin = MIN_OF_BOX;
    cnnBoxs[0].ymax = MAX_OF_BOX;

    ret = MppFrmResize(srcFrm, &resizeFrm, FRM_WIDTH, FRM_HEIGHT);  // resize 256*256
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for resize FAIL, ret=%x\n", ret);

    ret = FrmToOrigImg(&resizeFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "for Frm2Img FAIL, ret=%x\n", ret);

    ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[0]); // Crop the image to classfication network
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%x\n", ret);

    ret = CnnCalImg(self, &imgIn, resBuf, sizeof(resBuf) / sizeof((resBuf)[0]), &resLen);
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "cnn cal FAIL, ret=%x\n", ret);

    HI_ASSERT(resLen <= sizeof(resBuf) / sizeof(resBuf[0]));
    ret = CnnTrashClassifyFlag(resBuf, resLen, osdBuf, sizeof(osdBuf));
    SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnTrashClassifyFlag cal FAIL, ret=%x\n", ret);

    if (GetCfgBool("audio_player:support_audio", true)) {
        if (FdWriteMsg(g_stmChn.out, &resBuf[0], sizeof(RecogNumInfo)) != sizeof(RecogNumInfo)) {
            SAMPLE_PRT("FdWriteMsg FAIL\n");
        }
    }

    /*
     * 仅当计算结果与之前计算发生变化时，才重新打OSD输出文字
     * Only when the calculation result changes from the previous calculation, re-print the OSD output text
     */
    if (strcmp(osdBuf, prevOsd) != 0) {
        HiStrxfrm(prevOsd, osdBuf, sizeof(prevOsd));
        HI_OSD_ATTR_S rgn;
        TxtRgnInit(&rgn, osdBuf, TXT_BEGX, TXT_BEGY, ARGB1555_YELLOW2); // font width and heigt use default 40
        OsdsSetRgn(g_osdsTrash, g_osd0Trash, &rgn);
        /*
         * 用户向VPSS发送数据
         * User sends data to VPSS
         */
        // ret = HI_MPI_VPSS_SendFrame(0, 0, srcFrm, 0);
        // if (ret != HI_SUCCESS) {
        //     SAMPLE_PRT("Error(%#x), HI_MPI_VPSS_SendFrame failed!\n", ret);
        // }
    }

    IveImgDestroy(&imgIn);
    MppFrmDestroy(&resizeFrm);

    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

    
