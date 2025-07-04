#include "common.h"
#include "runtime.h"
#include "vulkan/meshVK.h"
#include "vulkan/rendererVK.h"
#include "vulkan/deviceVK.h"
#include "vulkan/utilsVK.h"

using namespace MiniEngine;

//#define RTX


MeshVK::MeshVK( const Runtime& i_runtime, const std::string& i_path, const std::vector<uint32_t> i_indices, const std::vector<Vertex> i_vertices ) :
    m_runtime       ( i_runtime   ),
    m_path          ( i_path      ),
    m_indices       ( i_indices   ),
    m_vertices      ( i_vertices  ),
    m_indices_buffer( VK_NULL_HANDLE ),
    m_data_buffer   ( VK_NULL_HANDLE )
{

}


bool MeshVK::initialize()
{
    assert( !m_indices.empty() && !m_vertices.empty() );

    if( !m_indices.empty() )
    {
        createIndexBuffer();

        UtilsVK::setObjectName( m_runtime.m_renderer->getDevice()->getLogicalDevice(), (uint64_t) m_indices_buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Indices Buffer" );
        UtilsVK::setObjectTag ( m_runtime.m_renderer->getDevice()->getLogicalDevice(), (uint64_t) m_indices_buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, 0, m_path.size(), m_path.c_str() );
    }

    if( !m_vertices.empty() )
    {
        m_data_buffer = createVertexBuffer( m_vertices, m_data_memory );

        UtilsVK::setObjectName( m_runtime.m_renderer->getDevice()->getLogicalDevice(), (uint64_t) m_indices_buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Vertex Buffer"                  );
        UtilsVK::setObjectTag ( m_runtime.m_renderer->getDevice()->getLogicalDevice(), (uint64_t) m_indices_buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, 0, m_path.size(), m_path.c_str() );
    }
    
    //BLAS
    //UtilsVK::createBLAS(*m_runtime.m_renderer->getDevice(), m_data_buffer, m_indices_buffer, m_vertices, m_indices, m_blas_structure, m_blas_buffer, m_blas_memory );
#ifdef RTX
    createBLASBuffer();
#endif // BLAS Creation

    return true;
}


void MeshVK::shutdown()
{
    const RendererVK&  renderer = *m_runtime.m_renderer;

    if( m_indices_buffer )
    {
        vkDestroyBuffer( renderer.getDevice()->getLogicalDevice(), m_indices_buffer, nullptr );
        vkFreeMemory   ( renderer.getDevice()->getLogicalDevice(), m_indices_memory, nullptr );
    }

    if( m_data_buffer )
    {
        vkDestroyBuffer( renderer.getDevice()->getLogicalDevice(), m_data_buffer, nullptr );
        vkFreeMemory   ( renderer.getDevice()->getLogicalDevice(), m_data_memory, nullptr );
    }

    
#ifdef RTX
    if (m_blas_buffer)
    {
        vkDestroyAccelerationStructure(renderer.getDevice()->getLogicalDevice(), m_blas, nullptr);
        vkDestroyBuffer(renderer.getDevice()->getLogicalDevice(), m_blas_buffer, nullptr);
        vkFreeMemory(renderer.getDevice()->getLogicalDevice(), m_blas_memory, nullptr);
    }
#endif
}


void MeshVK::draw( VkCommandBuffer& i_command_buffer, const uint32_t i_instance_id )
{
    VkBuffer data_buffers[] = { m_data_buffer };
    VkDeviceSize offsets [] = { 0 };

    UtilsVK::beginRegion( i_command_buffer, m_path.c_str(), Vector4f( 0.0f, 0.0f, 1.0f, 1.0f ) );
    UtilsVK::insert( i_command_buffer, m_path.c_str(), Vector4f( 0.0f, 0.5f, 0.5f, 1.0f ) );

    vkCmdBindIndexBuffer( i_command_buffer, m_indices_buffer, 0, VK_INDEX_TYPE_UINT32 );
    vkCmdBindVertexBuffers( i_command_buffer, 0, 1, data_buffers, offsets );
    vkCmdDrawIndexed( i_command_buffer, static_cast<uint32_t>( m_indices.size() ), 1, 0, 0, i_instance_id );

    UtilsVK::endRegion( i_command_buffer );
}

VkBuffer MeshVK::createVertexBuffer( const std::vector<Vertex>& i_data, VkDeviceMemory& i_memory )
{
    VkBuffer staging_buffer, vertex_buffer;
    VkDeviceMemory staging_memory;

    size_t size = sizeof( Vertex )*i_data.size();

    UtilsVK::createBuffer( *m_runtime.m_renderer->getDevice(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_memory );

    void* data;
    vkMapMemory( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory, 0, size, 0, &data );
    memcpy( data, i_data.data(), size );
    vkUnmapMemory( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory );


    UtilsVK::createBuffer( *m_runtime.m_renderer->getDevice(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, i_memory );

    UtilsVK::copyBuffer( *m_runtime.m_renderer->getDevice(), staging_buffer, vertex_buffer, size );

    vkDestroyBuffer( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_buffer, nullptr );
    vkFreeMemory   ( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory, nullptr );

    return vertex_buffer;
}

void MeshVK::createIndexBuffer()
{
    VkBuffer staging_buffer;
    VkDeviceMemory staging_memory;

    size_t size = sizeof( uint32_t )*m_indices.size();

    UtilsVK::createBuffer( *m_runtime.m_renderer->getDevice(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_memory );

    void* data;
    vkMapMemory( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory, 0, size, 0, &data );
    memcpy( data, m_indices.data(), size );
    vkUnmapMemory( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory );


    UtilsVK::createBuffer( *m_runtime.m_renderer->getDevice(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_indices_buffer, m_indices_memory );

    UtilsVK::copyBuffer( *m_runtime.m_renderer->getDevice(), staging_buffer, m_indices_buffer, size );

    vkDestroyBuffer( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_buffer, nullptr );
    vkFreeMemory   ( m_runtime.m_renderer->getDevice()->getLogicalDevice(), staging_memory, nullptr );
}

void MeshVK::createBLASBuffer()
{
    if (m_vertices.empty() || m_indices.empty())
    {
        std::cerr << "Cannot create BLAS - no vertex or index data available" << std::endl;
        return;
    }

    // Clean up any existing BLAS resources
    if (m_blas_buffer)
    {
        vkDestroyBuffer(m_runtime.m_renderer->getDevice()->getLogicalDevice(), m_blas_buffer, nullptr);
        vkFreeMemory(m_runtime.m_renderer->getDevice()->getLogicalDevice(), m_blas_memory, nullptr);
        m_blas_buffer = VK_NULL_HANDLE;
        m_blas_memory = VK_NULL_HANDLE;
    }
    
    if (m_blas_structure != NULL)
    {
        vkDestroyAccelerationStructure(m_runtime.m_renderer->getDevice()->getLogicalDevice(), m_blas_structure, nullptr);
        m_blas_structure = VK_NULL_HANDLE;
    }

    // Create the BLAS using the helper function from UtilsVK
    UtilsVK::createBLAS(
        *m_runtime.m_renderer->getDevice(),
        m_data_buffer,          // Vertex buffer
        m_indices_buffer,       // Index buffer
        m_vertices,             // Vertex data
        m_indices,              // Index data
        m_blas_structure,       // Output BLAS structure
        m_blas_buffer,         // Output BLAS buffer
        m_blas_memory           // Output BLAS memory
    );

    // Set debug names and tags
    UtilsVK::setObjectName(
        m_runtime.m_renderer->getDevice()->getLogicalDevice(),
        (uint64_t)m_blas_structure,
        VK_DEBUG_REPORT_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR_EXT,
        "BLAS Structure"
    );

    UtilsVK::setObjectName(
        m_runtime.m_renderer->getDevice()->getLogicalDevice(),
        (uint64_t)m_blas_buffer,
        VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,
        "BLAS Buffer"
    );

    UtilsVK::setObjectTag(
        m_runtime.m_renderer->getDevice()->getLogicalDevice(),
        (uint64_t)m_blas_buffer,
        VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT,
        0,
        m_path.size(),
        m_path.c_str()
    );
}