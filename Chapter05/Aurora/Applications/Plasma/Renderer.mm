
#import <simd/simd.h>
#import <ModelIO/ModelIO.h>

#import "Renderer.h"

// Include header shared between C code here, which executes Metal API commands, and .metal files
#import "ShaderTypes.h"

#include "pch.h"
#include "Plasma.h"
extern Plasma* gpApp;

static const NSUInteger kMaxBuffersInFlight = 3;

@implementation Renderer
{
    dispatch_semaphore_t _inFlightSemaphore;
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;
    
    id<MTLRenderPipelineState> _renderPipelineState;
    
    id<MTLTexture> _texture[kMaxBuffersInFlight];
    
    uint8_t _inFlightIndex;
}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view
{
    self = [super init];
    if(self)
    {
        _device = view.device;
        _inFlightSemaphore = dispatch_semaphore_create(kMaxBuffersInFlight);
        _inFlightIndex = 0;
        [self _loadMetalWithView:view];
        gpApp->run(view);
    }

    return self;
}

- (void)_loadMetalWithView:(nonnull MTKView *)view
{
    /// Load Metal state objects and initialize renderer dependent view properties
    
    view.depthStencilPixelFormat = MTLPixelFormatInvalid;
    view.colorPixelFormat = MTLPixelFormatRGBA16Float;
    view.sampleCount = 1;
//    view.layer.preferredDynamicRange = @"";
    
    _commandQueue = [_device newCommandQueue];
    
    id<MTLLibrary> library = [_device newDefaultLibrary];
    
    MTLRenderPipelineDescriptor* psoDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    psoDescriptor.label                            = @"Pass Through PSO";
    psoDescriptor.rasterSampleCount                = view.sampleCount;
    psoDescriptor.colorAttachments[0].pixelFormat  = view.colorPixelFormat;
    psoDescriptor.depthAttachmentPixelFormat       = view.depthStencilPixelFormat;
    psoDescriptor.vertexDescriptor                 = nil;
    psoDescriptor.vertexFunction                   = [library newFunctionWithName:@"pass_through_vertex"];
    psoDescriptor.fragmentFunction                 = [library newFunctionWithName:@"pass_through_fragment"];

    NSError* error = nil;
    _renderPipelineState = [_device newRenderPipelineStateWithDescriptor:psoDescriptor error:&error];
    if(!_renderPipelineState) {
        os_log(OS_LOG_DEFAULT, "Failed to create render pipeline state (%@): %@", psoDescriptor.label, error.description);
    }
    
    [library release];
    [psoDescriptor release];
}

- (void)_updateGameState
{
    gpApp->update();
}

- (void)drawInMTKView:(nonnull MTKView *)view
{
    /// Per frame updates here

    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"MyCommand";
    
    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^([[maybe_unused]] id<MTLCommandBuffer> buffer) {
        dispatch_semaphore_signal(block_sema);
    }];

    // run a frame and copy the result
    // TODO: we should access the texture directly, not copy the data out into another texture.
    [self _updateGameState];
    _texture[_inFlightIndex] = gpApp->getRenderImage();
    // Delay getting the currentRenderPassDescriptor until we absolutely need it to avoid
    //   holding onto the drawable and blocking the display pipeline any longer than necessary
    MTLRenderPassDescriptor* renderPassDescriptor = view.currentRenderPassDescriptor;

    if(renderPassDescriptor != nil) {

        id <MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"MyRenderEncoder";
        
        [renderEncoder setRenderPipelineState:_renderPipelineState];
        [renderEncoder setFragmentTexture:_texture[_inFlightIndex] atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:view.currentDrawable];
    }

    _inFlightIndex = (_inFlightIndex + 1) % kMaxBuffersInFlight;
    
    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    // TODO: Respond to drawable size or orientation changes here
}

@end
